
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import optax
from diffrax import diffeqsolve, ODETerm, SaveAt, Dopri5, Tsit5
import matplotlib.pyplot as plt
from time import time


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
    m_a, b_a, k_a, m_b, b_b, k_b, mu, drv, drv_args = args
    xa1, xa2, xb1, xb2 = x
    mu_a = mu(b_a, xa2)
    mu_b = mu(b_b, xb2)
    
    F0, omega = drv_args
    F = drv(F0, omega, t)

    xa1_t = xa2
    xa2_t = (-(k_a * xa1) + k_b * (xb1 - xa1) - mu_a * xa2 + F) / m_a

    xb1_t = xb2
    xb2_t = (-k_b * (xb1 - xa1) - mu_b * xb2) / m_b

    return xa1_t, xa2_t, xb1_t, xb2_t

def get_data(domain, size):
    term = ODETerm(coupled_vector_field)
    solver = Tsit5()
    t0 = 0
    t1 = domain
    dt0 = 0.1
    x0 = 0, 0, 1, 0 # x1, x1_t, x2, x2_t
    damping = lambda b, x_t: b * jnp.sign(x_t) * (x_t ** 2)
    driving = lambda F0, omega, t: F0 * jnp.cos(omega * t)
    #damping = lambda b, x_t: b * x_t
    drv_args = 10, 2 # F0, omega
    args = 70, 10, 250, 70, 10, 250, damping, driving, drv_args # m1, b1, k1, m2, b2, k2, damping
    saveat = SaveAt(ts=jnp.linspace(t0, t1, size))
    sol = diffeqsolve(term, solver, t0, t1, dt0, x0, args=args, saveat=saveat)

    consts = (x0, args)

    return sol.ts, sol.ys, consts


## Neural network definition:

class FNN(eqx.Module):
    layers: list
    bias: jnp.ndarray
    b: jnp.float32
    #mu: jnp.float32
    m1: jnp.float32
    k1: jnp.float32
    m2: jnp.float32
    k2: jnp.float32
    #mu2


    def __init__(self, in_size, out_size, hidden_size, *, key):
        key1, key2, key3, key4, key5, b_key,init_key = jax.random.split(key, 7)

        self.layers = [eqx.nn.Linear(in_size, hidden_size, key=key1),
                       eqx.nn.Linear(hidden_size, hidden_size, key=key3),
                       eqx.nn.Linear(hidden_size, hidden_size, key=key4),
                       #eqx.nn.Linear(hidden_size, hidden_size, key=key5),
                       eqx.nn.Linear(hidden_size, out_size, key=key2)]
        
        self.b = jnp.float32(0)
        #self.mu = jnp.float32(28.5)
        self.k1 = jnp.float32(250)
        self.m1 = jnp.float32(70)
        self.m2 = jnp.float32(70)
        self.k2 = jnp.float32(250)

        self.bias = jnp.zeros(out_size)

        
    def __call__(self,t):
        t = jnp.array(jnp.squeeze(t), ndmin=1)
        for layer in self.layers[:-1]:
            t = layer(t)
            t = jax.nn.tanh(t)
        out = (self.layers[-1](t)[0], self.layers[-1](t)[1])
        return out




## Main:

if True:
    dataset_size=1000
    pred_size = 0
    domain = 10
    collocation_step = 5
    learning_rate=2e-3
    steps=25000
    seed = jnp.int32(5492)

    model_key, test_key = jax.random.split(jax.random.PRNGKey(seed), 2)

    # Data acquisition:
    t, x, consts =  get_data(domain, dataset_size + pred_size) # t and y are NOT arrays yet

    initial_conds, ode_params = consts
    t_train = jnp.array(t[:dataset_size])
    t_phys = jnp.array(t[::5])
    x1_train = jnp.array(x[0][:dataset_size])
    x1_test = jnp.array(x[0])
    x2_test = jnp.array(x[2])
    t_test = jnp.array(t)

    # Model instantiatation:
    model = FNN(in_size=1, out_size=2, hidden_size=32, key=model_key) #ode_params initialization not implemented
    # Optimizer instatiation & initialization:
    optim = optax.adam(learning_rate, b1=0.9, b2=0.999)
    opt_state = optim.init(model)


    # Standard MSE loss function:
    @eqx.filter_value_and_grad
    def mse_loss(model, x, y):
         pred_y = jax.vmap(model)(x)
         pred_y = jnp.squeeze(pred_y)
         return jnp.mean((pred_y - y)**2)


    # PINN loss function:
    def PI_loss(model, t_dat, x1, t_phys, consts):
         
        initial_conds, ode_params = consts
        m1, b1, k1, m2, b2, k2, mu, F, F_args = ode_params
        F0, omega = F_args
        

        # forward pass:        
        pred_x1, pred_x2 = jax.vmap(model)(t_dat)
        #pred_x = jax.vmap(jnp.squeeze)(pred_x)

        # derivatives:
        x1_phys, x2_phys = jax.vmap(model)(t_phys)
        #y_phys = jax.vmap(jnp.squeeze)(y_phys)
        #x1_phys = x_phys[:][0]
        #x2_phys = x_phys[:][1]

        dydt = jax.jacrev(model)
        d2ydt2 = jax.jacrev(dydt)

        x1_t, x2_t = jax.vmap(dydt)(jnp.squeeze(t_phys))
        x1_tt, x2_tt = jax.vmap(d2ydt2)(jnp.squeeze(t_phys))

        # x1_t = x_t[:][0]
        # x2_t = x_t[:][1]
        # x1_tt = x_tt[:][0]
        # x2_tt = x_tt[:][1]  

        # NOT:
        # ddt = jax.grad(model)
        # dydt = jax.vmap(ddt(x))
        # d2ydt2 = jax.vmap(ddt(dydt))

        # data loss and physics loss:
        data_loss = jnp.mean(jnp.square(pred_x1 - x1))

        #f_pred = (1/(model.w)**2) * d2ydt2 + ((2*model.z)/model.w) * dydt + (y_phys) #- model.b)
        #f_pred = model.m * d2ydt2 + model.mu * dydt + model.k * (y_phys - 0) - 10#model.b)
        f_pred_x1 = m1 * x1_tt + k1 * x1_phys + k2 * (x1_phys - x2_phys) + mu(b1,x1_t) - F(F0, omega, t_phys) # residual evaluated for mass 1
        f_pred_x2 = m2 * x2_tt + k2 * (x2_phys - x1_phys) + mu(b2, x2_t) * x2_t # residual evaluated for mass 2
        phys_loss = jnp.mean(jnp.square(f_pred_x2)) #+ jnp.mean(jnp.square(f_pred_x1))

        dx1_0, dx2_0 = dydt(t[0])

        # initial conditions
        init_loss = jnp.square(initial_conds[0] - pred_x1[0]) + jnp.square(initial_conds[2] - pred_x2[0]) +\
                    jnp.square(initial_conds[1] - dx1_0) + jnp.square(initial_conds[3] - dx2_0)

        E = 0.99995

        return E * data_loss + (1-E) * phys_loss + 0.1*init_loss #+ loss_grad_loss#+ 0.1*pred_dy_t0**2

    @eqx.filter_grad
    def grad_calc(x1__x2):
        x1, x2 = x1__x2
        return x1, x2
    
    # Filtered JIT function for training loop:
    @eqx.filter_jit
    def filtered_func(model, t, x1, t_phys, consts, opt_state):
        # Compute loss:
        loss = PI_loss(model, t, x1, t_phys, consts)
        grads = jax.grad(PI_loss)(model, t, x1, t_phys, consts)

        # Optax SGD:
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state
    
    # @eqx.filter_jit
    # def mse_only(model, t, y, opt_state):
    #     # Compute loss:
    #     loss, grads = mse_loss(model, t, y)

    #     # Optax SGD:
    #     updates, opt_state = optim.update(grads, opt_state)
    #     model = eqx.apply_updates(model, updates)
    #     return loss, model, opt_state            
         

    #Training loop:
    for i in range(steps):
        if False:
             loss, model, opt_state = mse_only(model, T_data, X_data, opt_state)
        else:
            loss, model, opt_state = filtered_func(model, t_train, x1_train, t_phys, consts, opt_state)

        if np.mod(i,10) == 0:
            #lx1, lx2 = loss.item()
            print(f"step={i}, loss={loss.item()}, m1={model.m1}, k1={model.k1}")



    # Tesitng and data display:


    # Generate test data:
    x_pred = jax.vmap(model)(t_test)
    x1, x2 = x_pred 


    # Diplay test data:
    plt.grid(True)
    plt.plot(t_test, x1, label="x1_pred")
    plt.plot(t_test, x2, label="x2_pred")
    plt.plot(t_test, x2_test, label="x2_test")
    plt.plot(t_test, x1_test, label="x1_test")
    plt.axvline(x=(dataset_size/(dataset_size + pred_size)) * domain, linestyle='--', color='red') # draws vertical line where training data ends
    plt.legend()
    
    plt.xlabel('t')
    plt.ylabel('y')
    plt.savefig('Coupled_PINN/results/PINN_coupled_nox1phys_nonline_driven.png')
    plt.show()

