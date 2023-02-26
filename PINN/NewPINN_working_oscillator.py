
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import optax
import diffrax
import matplotlib.pyplot as plt
from time import time



def oscillator(d, w0, b0, A_mod, t):
    """Defines the analytical solution to the 1D underdamped harmonic oscillator problem. 
    Equations taken from: https://beltoforion.de/en/harmonic_oscillator/"""
    assert d < w0
    w = np.sqrt(w0**2-d**2)
    phi = np.arctan(-d/w)
#     A = A_mod * 1/(2*np.cos(phi))
    A = A_mod
    cosine = np.cos(phi+w*t)
    sine = np.sin(phi+w*t)
    exp = np.exp(-d*t)
    y  = exp*2*A*cosine + b0
    return y

#d, w0, b0, A_mod = 1.1, 20.2, 0.56, 0.15

# get the analytical solution over the full domain
#t_ana = np.linspace(0,1,500)
#x_ana = oscillator(d, w0, b0, A_mod, t_ana)


## Neural network definition:

class FNN(eqx.Module):
    layers: list
    bias: jnp.ndarray
    b: jnp.float32
    mu: jnp.float32
    m: jnp.float32
    k: jnp.float32
    # w: jnp.float32
    # z: jnp.float32

    def __init__(self, in_size, out_size, hidden_size, consts, *, key):
        key1, key2, key3, key4, key5, b_key,init_key = jax.random.split(key, 7)

        self.layers = [eqx.nn.Linear(in_size, hidden_size, key=key1),
                       eqx.nn.Linear(hidden_size, hidden_size, key=key3),
                       eqx.nn.Linear(hidden_size, hidden_size, key=key4),
                       #eqx.nn.Linear(hidden_size, hidden_size, key=key5),
                       eqx.nn.Linear(hidden_size, out_size, key=key2)]
        
        self.b = jnp.float32(0)
        self.mu = jnp.float32(28.5)
        self.k = jnp.float32(282.4)
        self.m = jnp.float32(70.8)
        # self.w = jnp.sqrt(consts[0]**2 - (consts[1] * consts[0])**2)
        # self.z = jnp.float32(consts[1])

        self.bias = jnp.zeros(out_size)

        
    def __call__(self,x):
        for layer in self.layers[:-1]:
            x = jax.nn.tanh(layer(jnp.array(x,ndmin=1)))
        return self.layers[-1](x)[0]



## Data generation:

def diffeq(t, consts):
        W0 = consts[0] # angular frequency
        Z = consts[1] # damping 
        A0 = consts[2] # amplitude
        b0 = consts[3] # vertical offset

        d = Z * W0 # damping
        W = jnp.sqrt(W0**2 - d**2) # natural frequency
        phi = jnp.arctan(-d/W) # phase angle


        # x(t) = 2A0 cos(wt + phi) e^(-Zt)
        return 2 * A0 * jnp.cos(W*t+phi) * jnp.exp(-d*t) + b0


def get_data(domain, dataset_size, pred_size):
    t = jnp.linspace(0, domain, (dataset_size + pred_size))
    t_data = t[:dataset_size]
    t_phys_sample = t[::5] # sampling points
    
    W0 = 2 # angular frequency
    Z = 0.1 # damping factor
    A0 = 0.5 # amplitude
    b0 = 0 # vertical offset

    diffeq_consts = [W0, Z, A0, b0]

    y = diffeq(t_data, diffeq_consts).astype(jnp.float32)
    y_phys_sample = diffeq(t_phys_sample, diffeq_consts).astype(jnp.float32)

    return jnp.array(t_data), y, jnp.array(t_phys_sample), jnp.array(y_phys_sample), diffeq_consts


## Main:

if True:
    dataset_size=400
    pred_size = 200
    domain = 8
    learning_rate=2e-3
    steps=100000
    seed = jnp.int32(5492)

    model_key, test_key = jax.random.split(jax.random.PRNGKey(seed), 2)

    # Data acquisition:
    t, y, t_phys_sample, y_phys_sample, consts = get_data(domain, dataset_size, pred_size) # t and y are NOT arrays yet
    
    # Model instantiatation:
    model = FNN(in_size=1, out_size=1, hidden_size=32, consts=consts, key=model_key)

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
    def PI_loss(model, x, y, x_phys_sample, consts):
         
        W0, Z = consts[:2]

        # forward pass:        
        pred_y = jax.vmap(model)(x)
        pred_y = jnp.squeeze(pred_y) 
        y = jnp.squeeze(y)

        # derivatives:
        y_phys = jax.vmap(model)(x_phys_sample)
        y_phys = jnp.squeeze(y_phys) 

        ddt = jax.grad(model)
        dt2 = jax.grad(ddt)

        dydt = jax.vmap(ddt)(jnp.squeeze(x_phys_sample))
        d2ydt2 = jax.vmap(dt2)(jnp.squeeze(x_phys_sample))
    
        # NOT:
        # ddt = jax.grad(model)
        # dydt = jax.vmap(ddt(x))
        # d2ydt2 = jax.vmap(ddt(dydt))

        # data loss and physics loss:
        data_loss = jnp.mean(jnp.square(pred_y - y))

        #f_pred = (1/(model.w)**2) * d2ydt2 + ((2*model.z)/model.w) * dydt + (y_phys) #- model.b)
        f_pred = model.m * d2ydt2 + model.mu * dydt + model.k * (y_phys - 0)#model.b)
        phys_loss = jnp.mean(jnp.square(f_pred))

        pred_y_t0 = pred_y[0] # = x0 = 1
        #pred_dy_t0 = dydt[0] # = v0 = 0
        boundary_loss = jnp.square(1 - pred_y_t0) #+ (pred_dy_t0)**2

        E = 0.9999995

        return E * data_loss + (1-E) * phys_loss + 0.1*boundary_loss #+ loss_grad_loss#+ 0.1*pred_dy_t0**2

    # Filtered JIT function for training loop:
    @eqx.filter_jit
    def filtered_func(model, t, y, x_phys_sample, opt_state):
        # Compute loss:
        loss = PI_loss(model, t, y, x_phys_sample, consts)
        grads = jax.grad(PI_loss,0)(model,t,y,x_phys_sample,consts)

        # Optax SGD:
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state
    
    @eqx.filter_jit
    def mse_only(model, t, y, opt_state):
        # Compute loss:
        loss, grads = mse_loss(model, t, y)

        # Optax SGD:
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state            
         

    # Training loop:
    for i in range(steps):
        if False:
             loss, model, opt_state = mse_only(model, T_data, X_data, opt_state)
        else:
            loss, model, opt_state = filtered_func(model, t, y, t_phys_sample, opt_state)

        if np.mod(i,10) == 0:
            print(f"step={i}, loss={loss.item()}, b={model.b}, m={model.m}, mu={model.mu}, k={model.k}")



    # Tesitng and data display:


    # Generate test data:
    t_test = jnp.linspace(0, domain, (dataset_size + pred_size))
    y_pred = jax.vmap(model)(t_test)


    # Diplay test data:

    plt.grid(True)
    plt.plot(t_test, y_pred)
    plt.plot(t_test, diffeq(t_test, [2, 0.1, 0.5, 0]))
    plt.axvline(x=(dataset_size/(dataset_size + pred_size)) * domain, linestyle='--', color='red') # draws vertical line where training data ends
    
    plt.xlabel('t')
    plt.ylabel('y')
    plt.savefig('PINN/results/PINN.png')
    plt.show()

