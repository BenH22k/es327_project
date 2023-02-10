import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import math
import optax


## Neural network definition:

class FNN(eqx.Module):
    layers: list
    bias: jnp.ndarray

    def __init__(self, in_size, out_size, hidden_size, *, key):
        key1, key2, key3 = jax.random.split(key, 3)

        self.layers = [eqx.nn.Linear(in_size, hidden_size, key=key1),
                       eqx.nn.Linear(hidden_size, hidden_size, key=key2),
                       eqx.nn.Linear(hidden_size, out_size, key=key3)]

        self.bias = jnp.zeros(out_size)
    
    def __call__(self,x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x) + self.bias


## Data:

def get_data(dataset_size):
    t = jnp.linspace(-1,1, dataset_size)
    y = t**2

    return t, y


## Main:

if True:
    dataset_size=100
    learning_rate=5e-3
    steps=1000
    seed = jnp.int32(5422)

    model_key, test_key = jax.random.split(jax.random.PRNGKey(seed), 2)

    # Data acquisition:
    t, y = get_data(dataset_size) # t and y are NOT arrays yet

    # Model instantiatation:
    #model = eqx.nn.MLP(in_size=1, out_size=1, width_size=8, depth=2, activation=jax.nn.tanh, key=model_key)
    model = FNN(in_size=1, out_size=1, hidden_size=8, key=model_key)


    # Optax optimiser:
    optim = optax.adam(learning_rate)
    opt_state = optim.init(model)

    # Loss function:
    @eqx.filter_value_and_grad
    def compute_loss(model, x, y):
         pred_y = jax.vmap(model)(x)
         pred_y = jnp.squeeze(pred_y)
         return jnp.mean((pred_y - y)**2)
   

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
        return loss, model, opt_state
    

    # Turns each element of vector into a 1d array (as required for forward pass):
    def format_vector(v):
        out = []
        for x in v:
            out.append(jnp.array(x,ndmin=1))
        return jnp.array(out)              
         
    # Training loop:
    #tf = jnp.reshape(len(t),1)
    tf = format_vector(t)

    #y = jnp.reshape(len(y),1)
    yf = jnp.array(y)

    for i in range(steps):
        loss, model, opt_state = filtered_func(model, tf, yf, opt_state)
        print(f"step={i}, loss={loss.item()}")


    # Generate test data:

    t_test = jnp.linspace(-1,1, 100)

    y_pred = jax.vmap(model)(format_vector(t_test))


    # Diplay test data:

    import matplotlib.pyplot as plt

    diffeq_consts_test = [2*jnp.pi, 0.1, 0, 1] # diffeq constants for test data comparison

    plt.grid(True)
    plt.plot(t_test, y_pred)
    plt.plot(t_test, t_test**2)
    
    plt.xlabel('t')
    plt.ylabel('y')
    plt.savefig('feedforwardNN/Results/resultsNewFNN.png')
    plt.show()

         






        

