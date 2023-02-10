import math

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import optax 
import equinox as eqx

def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jrandom.permutation(key, indices)
        (key,) = jrandom.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


def get_data(dataset_size, *, key):
    t = jnp.linspace(0, 2 * math.pi, 16)
    offset = jrandom.uniform(key, (dataset_size, 1), minval=0, maxval=2 * math.pi)
    x1 = jnp.sin(t + offset) / (1 + t)
    x2 = jnp.cos(t + offset) / (1 + t)
    y = jnp.ones((dataset_size, 1))

    half_dataset_size = dataset_size // 2
    x1 = x1.at[:half_dataset_size].multiply(-1)
    y = y.at[:half_dataset_size].set(0)
    x = jnp.stack([x1, x2], axis=-1)

    return x, y

class FNN(eqx.module):
    layers: list
    bias: jnp.ndarray

    def __init__(self, in_size, out_size, *, key):
        key1, key2, key3, key4 = jrandom.split(key, 4)
        self.list = [self.linear(in_size, 100, key1),
                     self.linear(100, 100, key2),
                     self.linear(100, 100, key3),
                     self.linear(100, out_size, key4)]
        self.bias = jnp.zeros(out_size)

    def __call__(self, input):
        for layer in self.layers[:-1]: # no activation on output layer
            x = jax.nn.tanh(layer(x))

        out, _ = lax.scan(input)
        return (out + self.bias)

def main(
    dataset_size=100,
    batch_size=1,
    learning_rate=3e-3,
    steps=100,
    depth=4,
    seed=5678,
):
    data_key, loader_key, model_key = jrandom.split(jrandom.PRNGKey(seed), 3)
    xs, ys = get_data(dataset_size, key=data_key)
    iter_data = dataloader((xs, ys), batch_size, key=loader_key)

    model = FNN(in_size=100, out_size=100, key=model_key)

    @eqx.filter_value_and_grad
    def compute_loss(model, x, y):
        pred_y = jax.vmap(model)(x)
        #return -jnp.mean(y * jnp.log(pred_y) + (1 - y) * jnp.log(1 - pred_y)) # binary cross entropy
        return jax.numpy.mean((y - pred_y) ** 2) # mse


    # Important for efficiency whenever you use JAX: wrap everything into a single JIT
    @eqx.filter_jit
    def make_step(model, x, y, opt_state):
        loss, grads = compute_loss(model, x, y)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    optim = optax.adam(learning_rate)
    opt_state = optim.init(model)
    for step, (x, y) in zip(range(steps), iter_data):
        loss, model, opt_state = make_step(model, x, y, opt_state)
        loss = loss.item()
        print(f"step={step}, loss={loss}")

    pred_ys = jax.vmap(model)(xs)
    num_correct = jnp.sum((pred_ys > 0.5) == ys)
    final_accuracy = (num_correct / dataset_size).item()
    print(f"final_accuracy={final_accuracy}")