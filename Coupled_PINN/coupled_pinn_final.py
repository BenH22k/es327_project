
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import optax
from diffrax import diffeqsolve, ODETerm, SaveAt, Dopri5, Tsit5
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from time import time
import csv

def coupled_vector_field(t, x, args):
    m_a, b_a, k_a, m_b, b_b, k_b, dmpa, dmpb, drv, drv_args = args
    xa1, xa2, xb1, xb2 = x
    mu_a = dmpa(b_a, xa2)
    mu_b = dmpb(b_b, xb2)
    
    F0, omega = drv_args
    F = drv(F0, omega, t)

    xa1_t = xa2
    xa2_t = (-(k_a * xa1) - k_b * (xa1 - xb1) - dmpa(b_a, xa2) - dmpb(b_b, xa2 - xb2) + F) / m_a

    xb1_t = xb2
    xb2_t = (-k_b * (xb1 - xa1) - dmpb(b_b, xb2 - xa2)) / m_b

    return xa1_t, xa2_t, xb1_t, xb2_t

def get_data(domain, size, key, driving, dampinga):
    term = ODETerm(coupled_vector_field)
    solver = Tsit5()
    t0 = 0
    t1 = domain
    dt0 = 0.05
    x0 = 1, 0, 0, 0 # x1, x1_t, x2, x2_t

    dampingb = lambda b, x_t:  b * x_t #+ b * x_t**2

    #driving = lambda F0, omega, t: 0
    drv_args = 0.5, 1 # F0, omega
    #args = m1, b1, k1, m2, b2, k2, ...
    args = 1, 0.25, 5, 1,0.25, 5, dampinga, dampingb, driving, drv_args
    saveat = SaveAt(ts=jnp.linspace(t0, t1, size))
    sol = diffeqsolve(term, solver, t0, t1, dt0, x0, args=args, saveat=saveat)

    assert len(sol.ts) == size

    consts = (x0, args)

    noise = 0.005 * jax.random.normal(key,jnp.array([size]),jnp.float32)

    noise = (noise[0], noise[1])
    x1_sol, dx1_sol, x2_sol, dx2_sol = sol.ys
    x1_noise =  x1_sol + noise[0]
    x_noise = (x1_noise, dx1_sol, x2_sol, dx2_sol, x1_sol)
    return sol.ts, x_noise, consts



## Neural network definition:

class FNN(eqx.Module):
    layers: list

    def __init__(self, in_size, out_size, hidden_size, depth, *, key):
        key1, key2, key3, key4, key5, key6 = jax.random.split(key, 6)

        self.layers = [eqx.nn.Linear(in_size, hidden_size, key=key1)]
        for i in range(depth):
            if i == 0:
                ky = key3
            elif i == 1:
                ky = key4
            elif i == 2:
                ky = key5
            elif i ==3:
                ky = key6

            self.layers.append(eqx.nn.Linear(hidden_size, hidden_size, key=ky))
        self.layers.append(eqx.nn.Linear(hidden_size, out_size, key=key2))

        
    def __call__(self,t):
        t = jnp.array(jnp.squeeze(t), ndmin=1)
        for layer in self.layers[:-1]:
            t = layer(t)
            t = jax.nn.tanh(t)
        out = (self.layers[-1](t)[0], self.layers[-1](t)[1])
        return out


# PINN loss function:
def PI_loss(model, t_dat, x1, t_phys, consts):

    initial_conds, ode_params = consts
    m1, b1, k1, m2, b2, k2, mu1, mu2, F, F_args = ode_params
    F0, omega = F_args
    
    # forward pass:        
    pred_x1, pred_x2 = jax.vmap(model)(t_dat)

    # derivatives:
    x1_phys, x2_phys = jax.vmap(model)(t_phys)

    dydt = jax.jacfwd(model)
    d2ydt2 = jax.jacfwd(dydt)

    x1_t, x2_t = jax.vmap(dydt)(jnp.squeeze(t_phys))
    x1_tt, x2_tt = jax.vmap(d2ydt2)(jnp.squeeze(t_phys))

    # data loss and physics loss:
    data_loss = jnp.mean(jnp.square(pred_x1 - x1))

    f_pred_x1 = m1 * x1_tt + k1 * x1_phys + k2 * (x1_phys - x2_phys) + mu1(b1,x1_t) + mu2(b2,x1_t-x2_t) - F(F0, omega, t_phys) # residual evaluated for mass 1
    f_pred_x2 = m2 * x2_tt + k2 * (x2_phys - x1_phys) + mu2(b2, x2_t-x1_t) # residual evaluated for mass 2
    phys_loss = jnp.mean(jnp.square(f_pred_x2)) + jnp.mean(jnp.square(f_pred_x1))

    # initial conditions
    dx1_0, dx2_0 = dydt(t_dat[0])

    init_loss = jnp.square(initial_conds[0] - pred_x1[0]) + jnp.square(initial_conds[2] - pred_x2[0]) +\
                jnp.square(initial_conds[1] - dx1_0) + jnp.square(initial_conds[3] - dx2_0)

    E = 0.999

    return E * data_loss + (1-E) * phys_loss + 0.1*init_loss #+ loss_grad_loss#+ 0.1*pred_dy_t0**2


def error_func(model, t, x1_test, x2_test):
    x1 , x2 = jax.vmap(model)(t)
    x1_loss = jnp.mean(jnp.square(x1_test - x1))
    x2_loss = jnp.mean(jnp.square(x2_test - x2))
    return x1_loss, x2_loss


# Filtered JIT function for training loop:
@eqx.filter_jit
def filtered_func(model, t, x1, x2_test, t_phys, consts, opt_state, optim):
    # Compute loss:
    loss = PI_loss(model, t, x1, t_phys, consts)
    grads = jax.grad(PI_loss)(model, t, x1, t_phys, consts)

    # Optax SGD:
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state



## Main:

def run_code(depth, hiddensize, domain, size, steps, driving, dampinga):

    dataset_size= size
    collocation_step = 3
    learning_rate=1e-3
    seed = jnp.int32(5492)

    model_key, data_key = jax.random.split(jax.random.PRNGKey(seed), 2)

    # Data acquisition:
    t, x, consts =  get_data(domain, dataset_size, data_key, driving, dampinga) 

    initial_conds, ode_params = consts
    t_train = jnp.array(t[:dataset_size])
    t_phys = jnp.array(t[::collocation_step])
    x1_train = jnp.array(x[0][:dataset_size])
    x1_test = jnp.array(x[4])
    x2_test = jnp.array(x[2])
    t_test = jnp.array(t)

    # Model instantiatation:
    model = FNN(in_size=1, out_size=2, hidden_size=hiddensize, depth=(depth-1), key=model_key) #ode_params initialization not implemented
    # Optimizer instatiation & initialization:
    optim = optax.adam(learning_rate, b1=0.9, b2=0.999)
    opt_state = optim.init(model)


    start_time = time()
    loss_y = []
    error_x1 = []
    error_x2 = []
    loss_interval = 100
    convstep = 0

    #for i in range(steps):
    i = 0
    while True:

        loss, model, opt_state = filtered_func(model, t_train, x1_train, x2_test, t_phys, consts, opt_state, optim)

        if np.mod(i,loss_interval) == 0:
            print(f"step={i}, loss={loss.item()}")
            loss_y.append(loss)

            ex1, ex2 = error_func(model, t_test, x1_test, x2_test)
            error_x1.append(ex1)
            error_x2.append(ex2)
        
        if i >= steps and loss < 1e-5: 
            print(f"Converged at {i} steps")
            convstep = i
            break

        if np.mod(i,100000) == 0 and i >50000:
            print("Stop training?")
            cond = ''
            cond = input()
            if cond == 'y' or cond == 'Y': break
        i += 1

    comp_time = time() - start_time
    print(f"computation time = {comp_time}")


    # Generate test data:
    x_pred = jax.vmap(model)(t_test)
    x1, x2 = x_pred 

    loss_x = []
    for i, y in enumerate(loss_y): loss_x.append((i * loss_interval))
    assert len(loss_x) == len(loss_y)
    loss_data = [loss_x, loss_y]

    error_x = []
    assert len(error_x1) == len(error_x2)
    for i, y in enumerate(error_x1): error_x.append((i * loss_interval))
    assert len(error_x) == len(error_x1)
    error_data = [error_x, error_x1, error_x2]

    data = [t_test, x1, x2, x1_test, x2_test]

    aux = [learning_rate, domain, collocation_step, loss_interval, comp_time, convstep]

    return data, loss_data, error_data, aux
        

if True:

    outputs = []
    losses = []
    errors = []
    auxs = []
    depth = 3
    width = 64
    steps = 25000
    driving = lambda F0, omega, t: F0 * jnp.cos(omega * t)
    dampinga = lambda b, x_t:  b * x_t + b * x_t**2
    for i in range(3):
        if i == 0:
            width = 32
        elif i == 1:
            width = 64
        elif i == 2:
            width = 128
        for j in range(1):
            if j == 0: domain = 10 * jnp.pi
            elif j == 1: domain = 3 * jnp.pi
            elif j == 2: domain = 4 * jnp.pi
            elif j == 3: domain = 5 * jnp.pi
            elif j == 4: domain = 6 * jnp.pi
            elif j == 5: domain = 7 * jnp.pi
            elif j == 6: domain = 8 * jnp.pi
            elif j == 7: domain = 9 * jnp.pi
            elif j == 8: domain = 10 * jnp.pi
            size = jnp.int16(round((domain/jnp.pi) * 80,0))
            print(f"w={width}, d={depth}, and T={domain/(jnp.pi)}")
            o, l, e, a = run_code(depth, width, domain, size, steps, driving, dampinga)
            outputs.append(o)
            losses.append(l)
            errors.append(e)
            auxs.append(a)

    with open('es327_project/Coupled_PINN/results/OUTPUT1.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(outputs)):
            for j in range(5):
                writer.writerow(outputs[i][j])


    with open('es327_project/Coupled_PINN/results/LOSS1.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(losses)):
            for j in range(2):
                writer.writerow(losses[i][j])        


    with open('es327_project/Coupled_PINN/results/ERROR1.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(errors)):
            for j in range(3):
                writer.writerow(errors[i][j])

    with open('es327_project/Coupled_PINN/results/AUX1.txt', 'w', newline='') as writer:
        for i in range(len(auxs)):
            writer.writelines(f"learing rate = {auxs[i][0]}, domain = {auxs[i][1]}, colloc interval = {auxs[i][2]}, sampling interval = {auxs[i][3]}, training time = {auxs[i][4]}, converged: {auxs[i][5]}\n")
    