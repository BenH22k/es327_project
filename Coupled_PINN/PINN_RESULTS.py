
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


## differential equation

# def vector_field(t, x, args):
#     wn, z = args
#     x1, x2 = x 
#     x1_t = x2
#     x2_t = - (2 * z * wn * x2) - ((wn**2) * x1)
#     return x1_t, x2_t


# #setup
# term = ODETerm(vector_field)
# solver = Dopri5()
# t0 = 0
# t1 = 10
# dt0 = 0.1
# x1_0 = 1, 0
# args = 2, 0.1

# saveat = SaveAt(ts=jnp.linspace(t0, t1, 1000))
# sol = diffeqsolve(term, solver, t0, t1, dt0, x1_0, args=args, saveat=saveat)

def coupled_vector_field(t, x, args):
    m_a, b_a, k_a, m_b, b_b, k_b, dmp, drv, drv_args = args
    xa1, xa2, xb1, xb2 = x
    mu_a = dmp(b_a, xa2)
    mu_b = dmp(b_b, xb2)
    
    F0, omega = drv_args
    F = drv(F0, omega, t)

    xa1_t = xa2
    xa2_t = (-(k_a * xa1) + k_b * (xb1 - xa1) - mu_a * xa2 + F) / m_a

    xb1_t = xb2
    xb2_t = (-k_b * (xb1 - xa1) - mu_b * xb2) / m_b

    return xa1_t, xa2_t, xb1_t, xb2_t

def get_data(domain, size, key):
    term = ODETerm(coupled_vector_field)
    solver = Tsit5()
    t0 = 0
    t1 = domain
    dt0 = 0.1
    x0 = 0, 0, 1, 0 # x1, x1_t, x2, x2_t
    #damping = lambda b, x_t: b * jnp.sign(x_t) * (x_t ** 2)
    #driving = lambda F0, omega, t: F0 * jnp.cos(omega * t)
    damping = lambda b, x_t: b * x_t
    driving = lambda F0, omega, t: 0
    drv_args = 10, 2 # F0, omega
    #args = 70, 10, 250, 70, 10, 250, damping, driving, drv_args # m1, b1, k1, m2, b2, k2, damping
    args = 1, 0.25, 10, 1, 0.25, 10, damping, driving, drv_args
    saveat = SaveAt(ts=jnp.linspace(t0, t1, size))
    sol = diffeqsolve(term, solver, t0, t1, dt0, x0, args=args, saveat=saveat)

    assert len(sol.ts) == size

    consts = (x0, args)

    noise = 0.005 * jax.random.normal(key,jnp.array([size]),jnp.float32)

    noise = (noise[0], noise[1])
    x1_sol, dx1_sol, x2_sol, dx2_sol = sol.ys
    x1_noise =  x1_sol #+ noise[0]
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
    m1, b1, k1, m2, b2, k2, mu, F, F_args = ode_params
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

    f_pred_x1 = m1 * x1_tt + k1 * x1_phys + k2 * (x1_phys - x2_phys) + mu(b1,x1_t) #- F(F0, omega, t_phys) # residual evaluated for mass 1
    f_pred_x2 = m2 * x2_tt + k2 * (x2_phys - x1_phys) + mu(b2, x2_t) * x2_t # residual evaluated for mass 2
    phys_loss = jnp.mean(jnp.square(f_pred_x2)) + jnp.mean(jnp.square(f_pred_x1))



    # initial conditions
    dx1_0, dx2_0 = dydt(t_dat[0])

    init_loss = jnp.square(initial_conds[0] - pred_x1[0]) + jnp.square(initial_conds[2] - pred_x2[0]) +\
                jnp.square(initial_conds[1] - dx1_0) + jnp.square(initial_conds[3] - dx2_0)

    E = 0.999

    return E * data_loss + (1-E) * phys_loss + 0.1*init_loss #+ loss_grad_loss#+ 0.1*pred_dy_t0**2


# Filtered JIT function for training loop:
@eqx.filter_jit
def filtered_func(model, t, x1, t_phys, consts, opt_state, optim):
    # Compute loss:
    loss = PI_loss(model, t, x1, t_phys, consts)
    grads = jax.grad(PI_loss)(model, t, x1, t_phys, consts)

    # Optax SGD:
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state



## Main:

def run_code(depth, hiddensize, steps):

    dataset_size=240
    pred_size = 0
    domain = 2 * jnp.pi
    collocation_step = 3
    learning_rate=1e-3
    seed = jnp.int32(5492)

    model_key, data_key = jax.random.split(jax.random.PRNGKey(seed), 2)

    # Data acquisition:
    t, x, consts =  get_data(domain, dataset_size + pred_size, data_key) 

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
    #loss_threshold = 
    # for i in range(steps):

    #     loss, model, opt_state = filtered_func(model, t_train, x1_train, t_phys, consts, opt_state, optim)

    #     if np.mod(i,10) == 0:
    #         print(f"step={i}, loss={loss.item()}")
    # print(f"computation time= {time() - start_time}")
    i = 0
    loss = 1
    while loss > 1e-5:
        i += 1
        loss, model, opt_state = filtered_func(model, t_train, x1_train, t_phys, consts, opt_state, optim)

        if np.mod(i,10) == 0:
            print(f"step={i}, loss={loss.item()}")

    print(f"computation time= {time() - start_time}")


    # Generate test data:
    x_pred = jax.vmap(model)(t_test)
    x1, x2 = x_pred 

    return t_test, x2, x2_test, x1_test, x1
        

if True:
    # subtitle_fontsize = 18
    # x_scaling = 1
    # fig, axs = plt.subplots(2)
    # fig.set_size_inches(12,15)
    # #fig.suptitle('Coupled Oscillator - Varying NN Width, $n$',fontsize=26)

    # # for i in range(1):
    # for j in range(2):
    #     axs[j].grid(True)
    #     axs[j].set_xlabel("t",fontsize=16)
    #     axs[j].set_ylabel("$x_2$",fontsize=16)
    #     axs[j].yaxis.set_major_locator(MaxNLocator(integer=True))
    #     #axs[i].set_ylim([-1.1,1.1])
    #     #axs[j].set_xlim([0,11])
    #     box = axs[j].get_position()
    #     axs[j].set_position([box.x0, box.y0 + box.height * 0.1,
    #                         box.width, box.height * 0.9])
        

    # for i in range(1):
    #     if i == 0:
    #         t_test, x2, x2_test = run_code(3,32,i)
    #         x2 = x2 * x_scaling
    #         x2_test = x2_test * x_scaling
    #         axs[0].plot(t_test,x2,label="$x_2$ Model Prediction", color='tab:blue')
    #         axs[0].scatter(t_test,x2_test,label="$x_2$ Test Data",color='r',marker='x')
    #         axs[0].set_title("$d=2$", fontsize=subtitle_fontsize)


    # #Put a legend below current axis
    # fig.legend(["$x_2$ Model Prediction", "$x_2$ Test Data"],loc='lower center', bbox_to_anchor=(0.5, 0.01),
    #         fancybox=True, shadow=True, ncol=2, fontsize=16)

    #fig.legend(["$x_2$ Model Prediction", "$x_2$ Test Data"], loc='lower center', ncol=1)
    # plt.savefig('Coupled_PINN/results/PINN_RESULTS.png')
    # plt.show()



    t_test, x2_test, x2, x1_test, x1 = run_code(3,32, 20000)
    plt.scatter(t_test,x2)
    plt.plot(t_test,x2_test)
    plt.savefig('Coupled_PINN/results/restest.png')

    with open('Coupled_PINN/results/RESULTS1.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(t_test)
        writer.writerow(x1_test)
        writer.writerow(x1)
        writer.writerow(x2_test)
        writer.writerow(x2)
        # writer.writerow(x2_III)

