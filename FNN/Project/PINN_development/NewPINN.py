
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import optax
import matplotlib.pyplot as plt



## Neural network definition:

class FNN(eqx.Module):
    layers: list
    bias: jnp.ndarray
    # can add ODE constants here to be trained along side NN

    def __init__(self, in_size, out_size, hidden_size, *, key):
        key1, key2, key3, key4, key5 = jax.random.split(key, 5)

        self.layers = [eqx.nn.Linear(in_size, hidden_size, key=key1),
                       eqx.nn.Linear(hidden_size, hidden_size, key=key3),
                       eqx.nn.Linear(hidden_size, hidden_size, key=key4),
                       #eqx.nn.Linear(hidden_size, hidden_size, key=key5),
                       eqx.nn.Linear(hidden_size, out_size, key=key2)]

        self.bias = jnp.zeros(out_size)
    
    def __call__(self,x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x)



## Data generation:

def diffeq1(t, consts): 
        Wn = consts[0]
        Z = consts[1]
        Phi = consts[2]
        H = consts[3]

        return H * (1 - ((jnp.exp(-Z * Wn * t)) / (jnp.sqrt(1 - Z**2))) * \
               (jnp.sin(Wn * jnp.sqrt(1 - Z**2) * t + Phi)))

def diffeq(t, consts):
        W0 = consts[0] # angular frequency
        Z = consts[1] # damping 
        A0 = consts[2] # amplitude

        d = Z * W0 # damping
        W = jnp.sqrt(W0**2 - d**2) # natural frequency
        phi = jnp.arctan(-d/W) # phase angle


        # x(t) = 2A0 cos(wt + phi) e^(-Zt)
        return 2 * A0 * (jnp.cos(W*t + phi) * jnp.exp(-d * t))



def get_data(dataset_size, pred_size):
    t = jnp.linspace(0, 1.5 * jnp.pi, (dataset_size + pred_size))
    t_data = t[:dataset_size]
    t_phys_sample = t[::5]
    
    W0 = 2 * jnp.pi # angular frequency
    Z = 0.1 # damping factor
    A0 = 1 # amplitude

    diffeq_consts = [W0, Z, A0]

    y = diffeq(t_data, diffeq_consts).astype(jnp.float32)
    y_phys_sample = diffeq(t_phys_sample, diffeq_consts).astype(jnp.float32)

    return t_data, y, t_phys_sample, y_phys_sample, diffeq_consts


## Main:

if True:
    dataset_size=200
    pred_size = 100
    learning_rate=5e-3
    steps=1000
    seed = jnp.int32(5422)

    model_key, test_key = jax.random.split(jax.random.PRNGKey(seed), 2)

    # Data acquisition:
    t, y, t_phys_sample, y_phys_sample, consts = get_data(dataset_size, pred_size) # t and y are NOT arrays yet

    # Model instantiatation:
    model = FNN(in_size=1, out_size=1, hidden_size=32, key=model_key)
    #model = eqx.nn.MLP(in_size=1, out_size=1, width_size=8, depth=2, activation=jax.nn.tanh, key=model_key) # Cannot initialize optimizer (for some reason)

    # Optimizer instatiation & initialization:
    optim = optax.adam(learning_rate)
    opt_state = optim.init(model)


    # # Loss function:
    # @eqx.filter_value_and_grad
    # def compute_loss(model, x, y):
    #     pred_y = jax.vmap(model)(x)
    #     pred_y = jnp.squeeze(pred_y)
    #     return jnp.mean((pred_y - y)**2)
    

    # PINN loss function:
    @eqx.filter_value_and_grad
    def PI_loss(model, x, y, x_phys_sample, consts, i):
         
        W0, Z = consts[:2]

        # forward pass:        
        pred_y = jax.vmap(model)(x)
        pred_y = jnp.squeeze(pred_y) 

        # derivatives:
        y_phys = jax.vmap(model)(x_phys_sample)
        print(y_phys.shape)
        #y_phys = jnp.squeeze(y_phys) 

        dydt = jax.vmap((jax.jacrev)(model, argnums = 0))(y_phys)
        dydt = jnp.squeeze(dydt,1)
        

        d2ydt2 = jax.vmap((jax.jacrev)(model, argnums = 0))(dydt)
        d2ydt2 = jnp.squeeze(d2ydt2)
        dydt = jnp.squeeze(dydt)
        y_phys = jnp.squeeze(y_phys)

        # data loss and physics loss:
        data_loss = jnp.mean((pred_y - y)**2)
        #phys_loss = jnp.mean((H * (((1/(Wn**2))*d2ydt2) + (((2*Z)/Wn)*dydt) + (y_phys - Phi))) ** 2) 
        f_pred = d2ydt2 + (2 * Z * W0) * dydt + (W0**2) * y_phys
        phys_loss = jnp.mean(f_pred ** 2)

        #pred_y0 = model(jnp.array([0]))[0]

        return ((1-E) * data_loss) + ((E) * phys_loss) #+ (pred_y0 - y[0]) ** 2
       

    # Filtered JIT function for training loop:
    @eqx.filter_jit
    def filtered_func(model, t, y, x_phys_sample, opt_state, i):
        # Compute loss:
        loss, grads = PI_loss(model, t, y, t, consts, i)

        # Optax SGD:
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state
    

    # Formats vectors for forward pass:
    def format_vector(v):
        out = []
        for x in v:
            out.append(jnp.array(x,ndmin=1))
        return jnp.array(out)              
         
    tf = format_vector(t)
    yf = jnp.array(y)
    t_phys_f = format_vector(t_phys_sample)


    # Training loop:
    for i in range(steps):
        loss, model, opt_state = filtered_func(model, tf, yf, t_phys_f, opt_state, i)
        print(f"step={i}, loss={loss.item()}")



    ## Tesitng and data display:


    # Generate test data:
    t_test = jnp.linspace(0, 1 * jnp.pi, 200)
    y_pred = jax.vmap(model)(format_vector(t_test))


    # Diplay test data:

    plt.grid(True)
    plt.plot(t_test, y_pred)
    plt.plot(t_test, diffeq(t_test, [2*jnp.pi, 0.1, 1]))
    
    plt.xlabel('t')
    plt.ylabel('y')
    plt.savefig('feedforwardNN/Results/results_PINNpredE1.png')
    plt.show()
