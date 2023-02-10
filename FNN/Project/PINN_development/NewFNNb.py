import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import math
import optax


## Neural network definition:

class FNN(eqx.Module):
    #mlp: eqx.Module
    mlp: eqx.nn.MLP
    bias: jnp.array
    
    def __init__(self, in_size, out_size, hidden_width, hidden_depth, *, key):
        #key_mlp = jax.random.split(key, 1)
        
        self.mlp = eqx.nn.MLP(in_size, out_size, hidden_width, hidden_depth, jax.nn.tanh, key=key)
        self.bias = jnp.zeros(out_size)

    # def __call__(self, x):
    #     return self.mlp(jnp.array(x, ndmin=1)) + self.bias


## Data:

def diffeq(t, consts): 
        Wn = consts[0]
        Z = consts[1]
        Phi = consts[2]
        H = consts[3]

        return H * (1 - ((jnp.exp(-Z * Wn * t)) / (jnp.sqrt(1 - Z**2))) * \
               (jnp.sin(Wn * jnp.sqrt(1 - Z**2) * t + Phi)))


def get_data_diffeq(dataset_size):
    t = jnp.linspace(0, 2 * jnp.pi, dataset_size)
    
    Wn = 2 * jnp.pi # natural frequency
    Z = 0.1 # damping factor
    Phi = 0
    H = 1

    diffeq_consts = [Wn, Z, Phi, H]

    y = diffeq(t, diffeq_consts).astype(jnp.float32)

    return t, y, diffeq_consts


## Main:

if True:
    dataset_size=100
    learning_rate=5e-3
    steps=1000
    seed = jnp.int32(5422)

    model_key, test_key = jax.random.split(jax.random.PRNGKey(seed), 2)

    # Data acquisition:
    t, y, consts = get_data_diffeq(dataset_size) # t and y are NOT arrays yet

    # Model instantiatation:
    model = eqx.nn.MLP(in_size=1, out_size=1, width_size=8, depth=2, activation=jax.nn.tanh, key=model_key)

    # Loss function:
    @eqx.filter_value_and_grad
    def compute_loss(model, x, y):
         pred_y = jax.vmap(model)(x)
         return jnp.mean((pred_y - y)**2) # this could be calculating wrong (maybe try lambda func?)


    def build_update_fn(optimizer: optax.adam):
         @eqx.filter_jit
         def update(model, opt_state):
              loss, grads = compute_loss(model, t, y)
              updates, opt_state = optimizer.update(grads, opt_state)
              model = eqx.apply_updates(model, updates)
              return model, opt_state
         return update


    def fit(
              optimizer: optax.adam,
              params: eqx.nn.MLP
    ) -> eqx.nn.MLP:
         update_fn = build_update_fn(optimizer)
         opt_state = optimizer.init(model)

         model, opt_state = update_fn(model, opt_state)
         return model
         


    # Optax optimiser:
    optim = optax.adam(learning_rate)
    opt_state = optim.init(model)

   
    

    # Testing:
    #################################################################
    # loss, grads = compute_loss(model, t, y)
    # print(f"{loss} and {grads}")
    #################################################################
    

    # Filtered function for training loop:

    @eqx.filter_jit
    def filtered_func(model, t, y, opt_state):
        # Compute loss:
        loss, grads = compute_loss(model, t, y)

        # Optax SGD:
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, grads, opt_state
    

    # Turns each element of vector into a 1d array (as required for forward pass):
    def format_vector(v):
        out = []
        for x in v:
            out.append(jnp.array(x,ndmin=1))
        return jnp.array(out)              
         
    # Training loop:

    for i in range(steps):
        loss, grads, opt_state = filtered_func(model, format_vector(t), jnp.array(y), opt_state)
        print(f"step={i}, loss={loss.item()}")


    # Generate test data:

    t_test = jnp.linspace(0, 2 * jnp.pi, 100)

    y_pred = jax.vmap(model)(t_test)


    # Diplay test data:

    import matplotlib.pyplot as plt

    diffeq_consts_test = [2*jnp.pi, 0.1, 0] # diffeq constants for test data comparison

    plt.grid(True)
    plt.plot(t_test, y_pred)
    plt.plot(t_test, diffeq(t_test, diffeq_consts_test))
    
    plt.xlabel('t')
    plt.ylabel('y')
    plt.savefig('feedforwardNN/Results/resultsNewFNN.png')
    plt.show()

         






        

