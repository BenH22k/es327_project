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

def get_data_sin(dataset_size, *, key):
    t = jnp.linspace(0, 1 * jnp.pi, 100)
    y = jnp.sin(t)
    return t, y

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
    
    # def __call__(self,x):
    #     for layer in self.layers[:-1]:
    #         x = jax.nn.tanh(layer(x))
    #     return (self.layers[-1](x)) + self.bias
    
    # batch compatible:
    def __call__(self,x_data):

        #x_data = jnp.expand_dims(x_data,0)

        for x in x_data:
            x = jnp.array([x])
            for layer in self.layers[:-1]:
                x = jax.nn.tanh(layer(x))
            x = self.layers[-1](x) + self.bias
            
        return x_data
        


def main(
    dataset_size=100,
    batch_size=10,
    learning_rate=3e-4,
    steps=100,
    #depth=3,
    seed=5678,
):
    data_key, loader_key, model_key = jax.random.split(jax.random.PRNGKey(seed), 3)
    t, y, consts = get_data_diffeq(dataset_size, key=data_key)

    # need every element as an array for model to run:
    # tj = []
    # yj = []

    # for i, elem in enumerate(t):
    #      tj.append(jnp.array([elem]))
    # for i, elem in enumerate(y):
    #     yj.append(jnp.array([elem]))

    # tj = jnp.array(tj)
    # yj = jnp.array(yj)
    tj = t
    yj = y
    ################################################

    iter_data = dataloader((tj, yj), batch_size, key=loader_key)

    model = FNN(in_size=1, out_size=1, hidden_size=32, key=model_key)

    # pure NN loss:
    # @eqx.filter_value_and_grad
    # def compute_loss(model, t, y):
    #     #pred_y = jax.vmap(model)(t)
    #     pred_y = model(t)
    #     return jax.numpy.mean((y - pred_y) ** 2) # mse

    # PINN loss:
    @eqx.filter_value_and_grad
    #@eqx.grad
    def PINN_loss(model, t, y, consts):

        Wn = consts[0]
        Z = consts[1]
        Phi = consts[2]
        
        t = jnp.expand_dims(t,0)
        t = jnp.swapaxes(t,0,1)
        pred_y = jax.vmap(model)(t)
        pred_y_t0 = model(jnp.array([jnp.float32([0])]))
        #pred_y_t15 = model(jnp.array([15]))

        def periodic_loss(pred_y):
            # minimise (1/Wn)(d2y/dt2) + (2Z/Wn)(dy/dt) + y

            pred_y_2d = pred_y
            # for elem in pred_y:
            #     pred_y_2d.append(jnp.array([elem]))
            # pred_y_2d = jnp.array(pred_y_2d)

            # minimise d2y/dt2 + dy/dt + y
            dydt = jax.vmap(jax.jacrev(model))(pred_y_2d)
            dydt = jnp.squeeze(dydt,1) # since an extra dimension was added 

            d2ydt2 = jax.vmap(jax.jacrev(model, argnums = 0))(dydt)

            #removes all the extra added dimensions:
            pred_y_2d = jnp.squeeze(pred_y_2d)
            dydt = jnp.squeeze(dydt)
            d2ydt2 = jnp.squeeze(d2ydt2)


            def loss_eqn(params):
                y_loc = params[0]
                ydot = params[1]
                ydotdot = params[2]
                return ((1/(Wn**2))*ydotdot) + (((2*Z)/Wn)*ydot) + y_loc
            
            loss_params = jnp.array([pred_y_2d, dydt, d2ydt2])
            out = jax.vmap(loss_eqn, 1)(loss_params)

            return out
        

        # only for t0 = 0 at the
        def boundary_loss():
            return 1 - pred_y_t0


        u_pred = pred_y
        u_loss = jnp.mean(((u_pred - y)**2)) # mean of MSEs

        f_pred = periodic_loss(pred_y)
        f_loss = jnp.mean(f_pred ** 2)

        b_pred = boundary_loss()
        b_loss = jnp.mean(b_pred ** 2)

        return jnp.float32(u_loss) #+ f_loss #+ b_loss


    @eqx.filter_jit
    def make_step(model, t, y, opt_state, consts):
        loss, grads = PINN_loss(model, t, y, consts)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    optim = optax.adam(learning_rate)
    opt_state = optim.init(model)

    consts_phys = consts # diffeq constants for physics informed part

    for step, (t, y) in zip(range(steps), iter_data):
        loss, model, opt_state = make_step(model, t, y, opt_state, consts_phys)
        loss = loss.item()
        print(f"step={step}, loss={loss}")


    res_t = []
    res_y = []


    t_test = jnp.linspace(0, 2 * jnp.pi, 100)
    for ts in t_test:
        pred_y = model(jnp.array([ts]))
        res_t.append(ts)
        res_y.append(pred_y)

    # results = [res_t,res_y]
    # print(results)

    import matplotlib.pyplot as plt

    diffeq_consts_test = [1*jnp.pi, 0.1, 0] # diffeq constants for test data generationdiff

    # # test training data:
    # plt.grid(True)
    # plt.plot(res_t, diffeq(t_test, diffeq_consts_phys))
    # plt.plot(res_t, diffeq(t_test, diffeq_consts_test))
    # ####################

    plt.grid(True)
    plt.plot(res_t, res_y)
    plt.plot(res_t, diffeq(t_test, diffeq_consts_test))
    
    plt.xlabel('t')
    plt.ylabel('y')
    plt.savefig('feedforwardNN/Results/resultsPINN3.png')
    plt.show()

    # import csv
    # with open('results4.csv', 'w') as f:
    #     mywriter = csv.writer(f, delimiter=',')
    #     mywriter.writerows(results)


if __name__ == "__main__":
    main()



