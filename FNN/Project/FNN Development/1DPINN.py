import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import math
import optax

def dataloader(arrays, batch_size, *, key):
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
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size

def get_data(dataset_size, *, key):
    t = jnp.linspace(0, 8 * jnp.pi, 100)
    y = jnp.sin(t)
    return t, y

class FNN(eqx.Module):
    layers: list
    bias: jnp.ndarray

    def __init__(self, in_size=1, out_size=1, hidden_size=1, *, key):
        key1, key2, key3, key4, key5, key6 = jax.random.split(key, 6)

        self.layers = [eqx.nn.Linear(in_size, hidden_size, key=key1),
                       #eqx.nn.Linear(hidden_size, hidden_size, key=key2),
                       #eqx.nn.Linear(hidden_size, hidden_size, key=key3),
                       #eqx.nn.Linear(hidden_size, hidden_size, key=key6),
                       eqx.nn.Linear(hidden_size, out_size, key=key4)]

        self.bias = jnp.zeros(out_size)
    
    def __call__(self,x):
        for layer in self.layers[:-1]:
            x = jax.nn.tanh(layer(x))
        return (self.layers[-1](x)) + self.bias


def main(
    dataset_size=100,
    batch_size=1,
    learning_rate=3e-3,
    steps=1000,
    #depth=3,
    seed=5678,
):
    data_key, loader_key, model_key = jax.random.split(jax.random.PRNGKey(seed), 3)
    t, y = get_data(dataset_size, key=data_key)
    iter_data = dataloader((t, y), batch_size, key=loader_key)

    model = FNN(in_size=1, out_size=1, hidden_size=200, key=model_key)

    # pure NN loss:
    # @eqx.filter_value_and_grad
    # def compute_loss(model, t, y):
    #     #pred_y = jax.vmap(model)(t)
    #     pred_y = model(t)
    #     return jax.numpy.mean((y - pred_y) ** 2) # mse

    # PINN loss:
    @eqx.filter_value_and_grad
    #@eqx.grad
    def PINN_loss(model, t, y):
        
        pred_y = model(t)

        def periodic_loss():
            # minimise d2y/dt2 + dy/dt + y
            dydt = jax.jacrev(model, argnums = 0)(pred_y)
            d2ydt2 = jax.jacrev(model, argnums = 0)(dydt)
            return d2ydt2 + dydt + pred_y
        
        def periodic_loss_sin():
            return - jnp.sin(t) + pred_y
        

        u_pred = pred_y
        u_loss = jnp.mean(((u_pred - y)**2))

        f_pred = periodic_loss_sin()
        f_loss = jnp.mean((f_pred**2))

        return u_loss #+ f_loss


    @eqx.filter_jit
    def make_step(model, t, y, opt_state):
        loss, grads = PINN_loss(model, t, y)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    optim = optax.adam(learning_rate)
    opt_state = optim.init(model)

    for step, (t, y) in zip(range(steps), iter_data):
        loss, model, opt_state = make_step(model, t, y, opt_state)
        loss = loss.item()
        print(f"step={step}, loss={loss}")

    # for step, (t,y) in enumerate(iter_data):
    #     ta = jnp.array([t])
    #     ya = jnp.array([y])
    #     loss, model, opt_state = make_step(model, ta, ya, opt_state)
    #     loss = loss.item()
    #     print(f"step={step}, loss={loss}")

    res_t = []
    res_y = []

    t_test = jnp.linspace(0, 8 * jnp.pi, 100)
    for ts in t_test:
        pred_y = model(jnp.array([ts]))
        res_t.append(ts)
        res_y.append(pred_y)

    results = [res_t,res_y]
    print(results)

    import matplotlib.pyplot as plt

    plt.grid(True)
    plt.plot(res_t, res_y)
    plt.plot(res_t, np.sin(res_t))
    
    plt.xlabel('t')
    plt.ylabel('y')
    plt.savefig('feedforwardNN/Results/resultsPINN.png')
    plt.show()

    # import csv
    # with open('results4.csv', 'w') as f:
    #     mywriter = csv.writer(f, delimiter=',')
    #     mywriter.writerows(results)


if __name__ == "__main__":
    main()



