import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import math
import optax

def dataloader(arrays, batch_size, *, key): # arrays contains the total array of t and the total array of y
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jax.random.permutation(key, indices)
        (key,) = jax.random.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays) # tuple of one t and one y value
            start = end
            end = start + batch_size


def diffeq(t, consts): 
        Wn = consts[0]
        Z = consts[1]
        Phi = consts[2]

        return (1 - ((jnp.exp(-Z * Wn * t)) / (jnp.sqrt(1 - Z**2))) * \
               (jnp.sin(Wn * jnp.sqrt(1 - Z**2) * t + Phi)))


def get_data_diffeq(dataset_size, *, key):
    t = jnp.linspace(0, 2 * jnp.pi, 100)
    
    Wn = 2 * jnp.pi # natural frequency
    Z = 0.1 # damping factor
    Phi = 0

    diffeq_consts = [Wn, Z, Phi]

    y = diffeq(t, diffeq_consts).astype(jnp.float32)

    return t, y, diffeq_consts

class FNN(eqx.Module):
    layers: list
    bias: jnp.ndarray

    def __init__(self, in_size=1, out_size=1, hidden_size=1, *, key):
        key1, key2, key3, key4, key5, key6 = jax.random.split(key, 6)

        self.layers = [eqx.nn.Linear(in_size, hidden_size, key=key1),
                       eqx.nn.Linear(hidden_size, hidden_size, key=key2),
                       eqx.nn.Linear(hidden_size, hidden_size, key=key3),
                       #eqx.nn.Linear(hidden_size, hidden_size, key=key6),
                       eqx.nn.Linear(hidden_size, out_size, key=key4)]

        self.bias = jnp.zeros(out_size)
    
    def __call__(self,x):
        for layer in self.layers[:-1]:
            x = jax.nn.tanh(layer(x))
        return (self.layers[-1](x)) + self.bias
            

if True:
#def main(
    dataset_size=100#,
    batch_size=99#,
    learning_rate=5e-3#,
    steps=5000#,
    seed=5678#,
#):
    data_key, loader_key, model_key = jax.random.split(jax.random.PRNGKey(seed), 3)
    t, y, consts = get_data_diffeq(dataset_size, key=data_key)

    iter_data = dataloader((t, y), batch_size, key=loader_key)

    model = FNN(in_size=1, out_size=1, hidden_size=32, key=model_key)
    print(model)

    # pure NN loss:
    @eqx.filter_value_and_grad
    def compute_loss(model, t, y):
        t = jnp.expand_dims(t,0)
        t = jnp.swapaxes(t,0,1)

        pred_y = jax.vmap(model)(t)
        #pred_y = model(t)
        return jax.numpy.mean((y - pred_y[:,0]) ** 2) # mse


    @eqx.filter_jit
    def make_step(model, t, y, opt_state):
        loss, grads = compute_loss(model, t, y)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    optim = optax.adam(learning_rate)
    opt_state = optim.init(model)

    for step, (t2, y2) in zip(range(steps), iter_data):
        #if jnp.ndim(t) == 1: t = jnp.swapaxes(jnp.expand_dims(t,0),0,1)
        loss, model, opt_state = make_step(model, t, y, opt_state)
        loss = loss.item()
        print(f"step={step}, loss={loss}")


    res_t = []
    res_y = []


    t_test = jnp.linspace(0, 2 * jnp.pi, 100)
    for ts in t_test:
        pred_y = model(jnp.array([ts]))
        res_t.append(jnp.float32(ts))
        res_y.append(jnp.float32(pred_y[0]))

    # results = [res_t,res_y]
    # print(results)

    import matplotlib.pyplot as plt

    diffeq_consts_test = [1*jnp.pi, 0.1, 0] # diffeq constants for test data generationdiff

    plt.grid(True)
    plt.plot(res_t, res_y)
    plt.plot(res_t, diffeq(t_test, diffeq_consts_test))
    
    plt.xlabel('t')
    plt.ylabel('y')
    plt.savefig('feedforwardNN/Results/resultsPINN3.png')
    plt.show()


# if __name__ == "__main__":
#     main()



