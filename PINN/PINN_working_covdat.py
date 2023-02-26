
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import optax
import diffrax
import matplotlib.pyplot as plt
from time import time

DTYPE='float32'


covid_world = np.loadtxt('project327/PINN/project/covid_world.dat')

days = np.arange(0,covid_world.shape[0])

def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')



covid_world_smooth = movingaverage(covid_world[:,1],7)
years = days/365

d1 = 345 
d2 = 695


plt.figure(figsize=(1100/72,400/72))
plt.plot((days[d1:d2]-d1)/365,covid_world[d1:d2,1]/1e06,color='blue',label='daily new cases')
plt.plot((days[d1:d2]-d1)/365,covid_world_smooth[d1:d2]/1e06,linewidth=2.0,color='red', label='smoothed daily new cases')
plt.xlabel('time [years]')
plt.ylabel('daily new cases worldwide')
plt.legend(loc='upper left',fontsize=14)

t_covid = days[d1:d2]-d1   # time array: days
y_covid = t_covid/365      # time array: years
x_covid = covid_world_smooth[d1:d2]/1e06  # normalize COVID numbers per 10^6 people 

# pick training data
t_data = y_covid[7:200:7]  # weekly data points work better than daily
x_data = x_covid[7:200:7]

# collocation points for enforcing ODE, minimizing residual
t_physics = y_covid[0::7]

t_data_jx = jnp.array(t_data, dtype=DTYPE)
x_data_jx = jnp.array(x_data, dtype=DTYPE)
t_physics_jx = jnp.array(t_physics, dtype=DTYPE)

T_data = jnp.reshape(t_data_jx[:], newshape=(t_data.shape[0],1))
X_data = jnp.reshape(x_data_jx[:], newshape=(x_data.shape[0],1))
T_r = jnp.reshape(t_physics_jx[:], newshape=(t_physics.shape[0],1))

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


d, w0, b0, A_mod = 1.1, 20.2, 0.56, 0.15

# get the analytical solution over the full domain
t_ana = np.linspace(0,1,500)
x_ana = oscillator(d, w0, b0, A_mod, t_ana)

plt.figure(figsize=(1100/72,400/72))
plt.plot((days[d1:d2]-d1)/365,covid_world[d1:d2,1]/1e06,color='blue',label='daily new cases')
plt.plot((days[d1:d2]-d1)/365,covid_world_smooth[d1:d2]/1e06,linewidth=2.0,color='red', label='smoothed daily new cases')
plt.plot(t_ana, x_ana, color='black', linewidth=2.0, label='analytical solution oscillator')
plt.xlabel('time [years]')
plt.ylabel('daily new cases worldwide')
plt.legend(loc='upper right',fontsize=14)


## Neural network definition:

class FNN(eqx.Module):
    layers: list
    bias: jnp.ndarray
    b: jnp.float32
    mu: jnp.float32
    m: jnp.float32
    k: jnp.float32
    # can add ODE constants here to be trained along side NN

    def __init__(self, in_size, out_size, hidden_size, *, key):
        key1, key2, key3, key4, key5, b_key,init_key = jax.random.split(key, 7)

        self.layers = [eqx.nn.Linear(in_size, hidden_size, key=key1),
                       eqx.nn.Linear(hidden_size, hidden_size, key=key3),
                       eqx.nn.Linear(hidden_size, hidden_size, key=key4),
                       #eqx.nn.Linear(hidden_size, hidden_size, key=key5),
                       eqx.nn.Linear(hidden_size, out_size, key=key2)]
        
        self.b = jnp.float32(0.56)
        self.mu = jnp.float32(2.2)
        self.k = jnp.float32(408)
        self.m = jnp.float32(1.0)
        
        # for layer in self.layers:
        #     layer = jax.random.normal(init_key, jnp.array(layer.out_features,ndmin=1), jnp.float32)

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
    b0 = 1 # vertical offset

    diffeq_consts = [W0, Z, A0, b0]

    y = diffeq(t_data, diffeq_consts).astype(jnp.float32)
    y_phys_sample = diffeq(t_phys_sample, diffeq_consts).astype(jnp.float32)

    return jnp.array(t_data), y, jnp.array(t_phys_sample), jnp.array(y_phys_sample), diffeq_consts


## Main:

if True:
    dataset_size=400
    pred_size = 200
    domain = 5
    learning_rate=2e-3
    steps=25000
    seed = jnp.int32(5492)

    model_key, test_key = jax.random.split(jax.random.PRNGKey(seed), 2)

    # Data acquisition:
    t, y, t_phys_sample, y_phys_sample, consts = get_data(domain, dataset_size, pred_size) # t and y are NOT arrays yet

    # Model instantiatation:
    model = FNN(in_size=1, out_size=1, hidden_size=32, key=model_key)

    # Optimizer instatiation & initialization:
    optim = optax.adam(learning_rate, b1=0.9, b2=0.999)
    opt_state = optim.init(model)


    # Standard MSE loss function:
    @eqx.filter_value_and_grad
    def mse_loss(model, x, y):
         pred_y = jax.vmap(model)(x)
         pred_y = jnp.squeeze(pred_y)
         return jnp.mean((pred_y - y)**2)


    def diff_eq(m, mu, k, b, y, dydt, d2ydt2):
        #return d2ydt2 + (2 * Z * W0) * dydt + (W0**2) * (y - b)
        return model.m * d2ydt2 + model.mu * dydt + model.k * (y - model.b)


    # PINN loss function:
    def PI_loss(model, x, y, x_phys_sample, consts):
         
        W0, Z = consts[:2]

        # forward pass:        
        pred_y = jax.vmap(model)(x[:,0:1])
        pred_y = jnp.squeeze(pred_y) 
        y = jnp.squeeze(y)

        # derivatives:
        y_phys = jax.vmap(model)(x_phys_sample)
        y_phys = jnp.squeeze(y_phys) 

        # def ODE_loss(t,u):
        #     u_t=lambda t: jax.grad(lambda t: u(t))(t)
        #     u_tt=lambda t: jax.grad(lambda t : jnp.squeeze(u_t(t)))(t)
        #     return model.m * jax.vmap(u_tt)(t) + model.mu * jax.vmap(u_t)(t) + model.k * (jax.vmap(u)(t) - model.b)

        # ufunc = lambda t: model(t)
        # phys_loss = jnp.mean(ODE_loss(x_phys_sample,ufunc)**2)
        ddt = jax.grad(model)
        dt2 = jax.grad(ddt)

        dydt = jax.vmap(ddt)(jnp.squeeze(x_phys_sample))
        d2ydt2 = jax.vmap(dt2)(jnp.squeeze(x_phys_sample))

        # data loss and physics loss:
        data_loss = jnp.mean(jnp.square(pred_y - y))

        f_pred = model.m * d2ydt2 + model.mu * dydt + model.k * (y_phys - model.b)
        phys_loss = jnp.mean(jnp.square(f_pred))
        #f_pred = d2ydt2 + (2 * Z * W0) * dydt + (W0**2) * y_phys

        #phys_loss = jnp.mean(jnp.square(f_pred))

        pred_y_t0 = pred_y[0] # = x0 = 1
        #pred_dy_t0 = dydt[0] # = v0 = 0

        boundary_loss = jnp.square(2 - pred_y_t0) #+ (pred_dy_t0)**2

        E = 0.9999995

        return E * data_loss + (1-E) * phys_loss #+ 0.1*boundary_loss #+ loss_grad_loss#+ 0.1*pred_dy_t0**2

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
            loss, model, opt_state = filtered_func(model, T_data, X_data, T_r, opt_state)

        if np.mod(i,10) == 0:
            print(f"step={i}, loss={loss.item()}, b={model.b}")



    ## Tesitng and data display:


    # # Generate test data:
    # t_test = jnp.linspace(0, domain, (dataset_size + pred_size))
    # y_pred = jax.vmap(model)(t_test)


    # # Diplay test data:

    # plt.grid(True)
    # plt.plot(t_test, y_pred)
    # plt.plot(t_test, diffeq(t_test, [2, 0.1, 0.5, 1]))
    # plt.plot(T)
    # plt.axvline(x=(dataset_size/(dataset_size + pred_size)) * domain, linestyle='--', color='red') # draws vertical line where training data ends
    
    # plt.xlabel('t')
    # plt.ylabel('y')
    # plt.savefig('project327/PINN/results/PINN.png')
    # plt.show()


    def plot_solution(model, **kwargs):
        
        n=d2-d1
        t_pred = np.reshape(np.linspace(0,n,n),(n,1))/365
        x_pred = jax.vmap(model)(t_pred)

        # plot prediction
        fig, ax2 = plt.subplots(figsize=(700/72,500/72))
        ax2.set_title('PINN COVID')
        ax2.scatter(t_data,x_data,s=300,color="tab:orange", alpha=1.0,marker='.') #observed data points 
        ax2.plot((days[d1:d2]-d1)/365,covid_world[d1:d2,1]/1e06,color='blue')
        ax2.plot((days[d1:d2]-d1)/365,covid_world_smooth[d1:d2]/1e06,linewidth=2.0,color='red')
        ax2.plot(t_pred,x_pred,color="black",linewidth=3.0,linestyle="--")
        ax2.legend(('training data','daily cases','daily cases smooth','model prediction'), loc='upper right',fontsize=14)
        ax2.set_xlabel('time [years]',fontsize=14)
        ax2.set_ylabel('daily new cases [per 10^6]]',fontsize=14)
        
        plt.savefig('Cov_PINN_a1.pdf',bbox_inches='tight')
        
        return ax2
        
    def plot_loss_history(self, ax=None):
        if not ax:
            fig = plt.figure(figsize=(700/72,500/72))
            ax = fig.add_subplot(111)
        ax.semilogy(range(len(self.hist)), self.hist,'k-')
        ax.set_xlabel('$n_{epoch}$',fontsize=18)
        ax.set_ylabel('$\\phi^{n_{epoch}}$',fontsize=18)
        return ax

    def plot_loss_and_param(self, axs=None):

        color_mu = 'tab:blue'
        color_k = 'tab:red'
        color_b = 'tab:green'

        fig = plt.figure(figsize=(1200/72,800/72))
        gs = fig.add_gridspec(2, 2)
        
        ax1 = plt.subplot(gs[0, 0])
        ax1 = self.plot_loss_history(ax1)
        
        ax2 = plt.subplot(gs[0, 1])
        ax2.plot(range(len(self.hist)), self.model.mu_list,'-',color=color_mu)
        ax2.set_ylabel('$\\mu^{n_{epoch}}$', color=color_mu, fontsize=18)
        ax2.set_xlabel('$n_{epoch}$',fontsize=18)
        
        ax3 = plt.subplot(gs[1, 0])
        ax3.plot(range(len(self.hist)), self.model.k_list,'-',color=color_k)
        ax3.set_ylabel('$k^{n_{epoch}}$', color=color_k, fontsize=18)
        ax3.set_xlabel('$n_{epoch}$',fontsize=18)
        
        ax4 = plt.subplot(gs[1, 1])
        ax4.plot(range(len(self.hist)), self.model.b_list,'-',color=color_b)
        ax4.set_ylabel('$b^{n_{epoch}}$', color=color_b, fontsize=18)
        ax4.set_xlabel('$n_{epoch}$',fontsize=18)

        return (ax1,ax2,ax3,ax4)

    ax = plot_solution(model);
    #axs = plot_loss_and_param(model);
