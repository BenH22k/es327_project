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
    t = jnp.linspace(0, 2 * jnp.pi, 100)
    y = jnp.sin(t)
    return t, y

def dataloader2(t, y, key):
    iter_data = []
    for i in range(len(t)):
        iter_data.append((t[i],y[i]))
    return iter_data


class FNN(eqx.Module):
    layers: list
    bias: jnp.ndarray

    def __init__(self, in_size=100, out_size=100, *, key):
        key1, key2, key3, key4 = jax.random.split(key, 4)

        self.layers = [eqx.nn.Linear(in_size, 100, key=key1),
                       eqx.nn.Linear(100, 100, key=key2),
                       #eqx.nn.Linear(100, 100, key=key3),
                       eqx.nn.Linear(100, out_size, key=key4)]

        self.bias = jnp.zeros(out_size)
    
    def __call__(self,x):
        for layer in self.layers[:-1]:
            x = jax.nn.tanh(layer(x))
        return self.layers[-1](x) + self.bias


def main(
    dataset_size=100,
    #batch_size=1,
    learning_rate=3e-3,
    steps=50,
    depth=4,
    seed=5678,
):
    data_key, loader_key, model_key = jax.random.split(jax.random.PRNGKey(seed), 3)
    t, y = get_data(dataset_size, key=data_key)
    #iter_data = dataloader((ys), batch_size, key=loader_key)
    iter_data = dataloader2(t, y, key=loader_key)

    model = FNN(in_size=100, out_size=100, key=model_key)

    @eqx.filter_value_and_grad
    def compute_loss(model, t, y):
        #pred_y = jax.vmap(model)(t)
        pred_y = model(t)
        return jax.numpy.mean((y - pred_y) ** 2) # mse

    @eqx.filter_jit
    def make_step(model, t, y, opt_state):
        loss, grads = compute_loss(model, t, y)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    optim = optax.adam(learning_rate)
    opt_state = optim.init(model)
    for i in range(steps):
        loss, model, opt_state = make_step(model, t, y, opt_state)
        loss = loss.item()
        print(f"step={i}, loss={loss}")

    #pred_y = jax.vmap(model)(t)
    pred_y = model(t)

    #testing
    t_test = jnp.linspace(0, 3 * jnp.pi, 100)
    pred_y = model(t_test)
    real_y = jnp.sin(t_test)

    #print(pred_y)
    pred_y = np.asarray(pred_y)
    results = [t, pred_y, real_y]

    import csv
    with open('results4.csv', 'w') as f:
        mywriter = csv.writer(f, delimiter=',')
        mywriter.writerows(results)

    import matplotlib.pyplot as plt

    plt.grid(True)
    plt.plot(t_test, pred_y)
    plt.plot(t_test, real_y)
    
    plt.xlabel('t')
    plt.ylabel('y')
    plt.savefig('results2d.png')
    plt.show()


if __name__ == "__main__":
    main()














# def compute_loss(model, x, y):
#     pred_y = jax.vmap(model)(x)
#     return jax.numpy.mean((y - pred_y) ** 2)

# in_size = 100
# out_size = 100

# x_key, y_key, model_key = jax.random.split(jax.random.PRNGKey(0), 3)
# x = jax.random.normal(x_key, (100, 100))
# y = jax.random.normal(y_key, (100, 100))
# model = FNN(in_size, out_size, key=model_key)
# grads = compute_loss(model, x, y)
# learning_rate = 0.1
# #model = jax.tree_util.tree_map(lambda m, g: m - learning_rate * g, model, grads)

# def make_step(model, x, y, opt_state):
#     loss, grads = compute_loss(model, x, y)
#     updates, opt_state = optim.update(grads, opt_state)
#     model = eqx.apply_updates(model, updates)
#     return loss, model, opt_state

# optim = optax.adam(learning_rate)
# opt_state = optim.init(model)
# for step, (x, y) in zip(range(steps), iter_data):
#     loss, model, opt_state = make_step(model, x, y, opt_state)
#     loss = loss.item()
#     print(f"step={step}, loss={loss}")

# pred_ys = jax.vmap(model)(xs)
# num_correct = jnp.sum((pred_ys > 0.5) == ys)
# final_accuracy = (num_correct / dataset_size).item()
# print(f"final_accuracy={final_accuracy}")