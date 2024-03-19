#%%
import time

import flax 
import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax  # https://github.com/deepmind/optax

# class Func(eqx.Module):
#     mlp: eqx.nn.MLP

#     def __init__(self, data_size, width_size, depth, *, key, **kwargs):
#         super().__init__(**kwargs)
#         self.mlp = eqx.nn.MLP(
#             in_size=data_size,
#             out_size=data_size,
#             width_size=width_size,
#             depth=depth,
#             activation=jnn.softplus,
#             key=key,
#         )

#     def __call__(self, t, y, args):
#         return self.mlp(y)
    

# class NeuralODE(eqx.Module):
#     func: Func

#     def __init__(self, data_size, width_size, depth, *, key, **kwargs):
#         super().__init__(**kwargs)
#         self.func = Func(data_size, width_size, depth, key=key)

#     def __call__(self, ts, y0):
#         solution = diffrax.diffeqsolve(
#             diffrax.ODETerm(self.func),
#             diffrax.Tsit5(),
#             t0=ts[0],
#             t1=ts[-1],
#             dt0=ts[1] - ts[0],
#             y0=y0,
#             stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
#             saveat=diffrax.SaveAt(ts=ts),
#         )
#         return solution.ys

import flax.linen as nn
import jax.numpy as jnp
import jax.random as random
from jax.example_libraries import optimizers, stax
from diffrax import diffeqsolve, ODETerm, Tsit5, PIDController, SaveAt

class FuncFlax(nn.Module):
    data_size: int
    width_size: int
    depth: int

    @nn.compact
    def __call__(self, y):
        # Define an MLP using Flax
        # Initialize weights using a normal distribution
        # Note: Flax handles parameter initialization differently than Equinox
        mlp = stax.serial(
            *[stax.serial(stax.Dense(self.width_size), stax.Softplus()) for _ in range(self.depth)],
            stax.Dense(self.data_size),
            stax.Softplus()
        )
        return mlp(y)

class NeuralODEFlax(nn.Module):
    data_size: int
    width_size: int
    depth: int

    def setup(self):
        self.func = FuncFlax(data_size=self.data_size, width_size=self.width_size, depth=self.depth)

    def __call__(self, ts, y0, rng_key):
        def ode_func(t, y, args):
            return self.func(y)  # Adjust based on how you handle RNGs in FuncFlax if necessary
        
        solution = diffeqsolve(
            ODETerm(ode_func),
            Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            stepsize_controller=PIDController(rtol=1e-3, atol=1e-6),
            saveat=SaveAt(ts=ts),
        )
        return solution.ys

    
#%%
def _get_data(ts, *, key):
    y0 = jr.uniform(key, (2,), minval=-2, maxval=2)

    def f(t, y, args):
        mu = 3.0
        x, v = y
        dxdt = v
        dvdt = mu * (1 - x**2) * v - x
        return jnp.stack([dxdt, dvdt], axis=-1)

    solver = diffrax.Tsit5()
    dt0 = 0.1
    saveat = diffrax.SaveAt(ts=ts)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(f), solver, ts[0], ts[-1], dt0, y0, saveat=saveat
    )
    ys = sol.ys
    return ys


def get_data(dataset_size, *, key):
    ts = jnp.linspace(0, 10, 100)
    key = jr.split(key, dataset_size)
    ys = jax.vmap(lambda key: _get_data(ts, key=key))(key)
    return ts, ys

#%%
dataset_size = 50
#plot the data
ts, ys = get_data(dataset_size, key=jr.PRNGKey(1234))
for y in ys:
    plt.plot(ts, y[:, 0])
plt.show()
#%%
#plot the second dimension
for y in ys:
    plt.plot(ts, y[:, 1])
plt.show()
## %%  

# import numpy as np
# import jax.numpy as jnp
# from scipy.integrate import solve_ivp

# # Van der Pol oscillator parameters
# mu = 3.0  # Choose a value to ensure non-linear oscillation and limit cycle behavior

# # Van der Pol system differential equations
# def van_der_pol_system(t, state):
#     x, y = state
#     dxdt = y
#     dydt = mu * (1 - x**2) * y - x
#     return [dxdt, dydt]

# # Time span for the simulation
# t_span = (0, 50)
# t_eval = np.linspace(*t_span, 10000)

# # Initial conditions
# initial_conditions = np.random.rand(1000, 2) * 4 - 2  # Random initial conditions within a reasonable range

# # Simulate the Van der Pol system for different initial conditions
# trajectories = np.array([solve_ivp(van_der_pol_system, t_span, ic, t_eval=t_eval).y for ic in initial_conditions])

# # Adjust the code for handling the 2D trajectories instead of 3D as with the Lorenz system
# # Convert to JAX arrays for compatibility
# trajectories_jax = jnp.array(trajectories)

# # Determine the split index for training and testing datasets
# split_idx = int(trajectories_jax.shape[0] * 0.8)

# # Split the dataset and adjust for 2D data
# train_batches = train_data = jnp.transpose(trajectories_jax[:split_idx], axes=(0, 2, 1))
# test_batches = test_data = jnp.transpose(trajectories_jax[split_idx:], axes=(0, 2, 1))


#%%
def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jr.permutation(key, indices)
        (key,) = jr.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size

loader_key = jr.PRNGKey(1234)
_ys = dataloader((ys,), batch_size=32, key=loader_key)




#%%
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jr
from flax.training import train_state
import optax
from diffrax import diffeqsolve, ODETerm, Tsit5, PIDController, SaveAt
import time
from jax import vmap
from functools import partial

# Assuming `FuncFlax` and `NeuralODEFlax` are defined as in your previous code snippet.

batch_size = 32
width_size = 64
depth = 2
seed = 5678
plot = True
print_every = 100

key = jr.PRNGKey(seed)
data_key, model_key, loader_key = jr.split(key, 3)

# Assuming `ts` (time points) and `ys` (data batches) are defined somewhere
# dataset_size, length_size, data_size = ys.shape

class TrainState(train_state.TrainState):
    batch_stats: dict

def create_train_state(rng_key, model, learning_rate):
    params = model.init(rng_key, jnp.ones((batch_size, length_size)), jnp.ones((batch_size, data_size)), rng_key)['params']
    tx = optax.adam(learning_rate)
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

def compute_loss(model, params, ts, y0, ys, rng_key):
    preds = vmap(model.apply, in_axes=(None, 0, 0, None), out_axes=0)({'params': params}, ts, y0, rng_key)
    return jnp.mean((ys - preds) ** 2)

@jax.jit
def train_step(state, ts, y0, ys, rng_key):
    grads = jax.grad(compute_loss)(state.apply_fn, state.params, ts, y0, ys, rng_key)
    return state.apply_gradients(grads=grads)

# Initialize model and optimizer
model = NeuralODEFlax(data_size, width_size, depth)
learning_rate = 3e-3  # Initial learning rate
state = create_train_state(model_key, model, learning_rate)

# Example training loop
for lr, iters, length in zip(lr_strategy, steps_strategy, length_strategy):
    _ts = ts[: int(length_size * length)]
    _ys = ys[:, : int(length_size * length)]
    for iter in range(iters):
        start = time.time()
        # Example: Randomly select a batch from _ys for simplicity
        # In practice, use your dataloader or batching mechanism
        yi = _ys[:batch_size, :]
        y0 = yi[:, 0, :]
        state = train_step(state, _ts, y0, yi, model_key)
        end = time.time()
        if (iter % print_every) == 0 or iter == iters - 1:
            loss = compute_loss(state.apply_fn, state.params, _ts, y0, yi, model_key)
            print(f"Iteration: {iter}, Loss: {loss}, Computation time: {end - start}")










#%%
# dataset_size=256,
batch_size=32

width_size=64
depth=2
seed=5678
plot=True
print_every=100

key = jr.PRNGKey(seed)
data_key, model_key, loader_key = jr.split(key, 3)

# ts, ys = t_eval, train_batches
dataset_size, length_size, data_size = ys.shape

model = NeuralODEFlax(data_size, width_size, depth)

# Training loop like normal.
#
# Only thing to notice is that up until step 500 we train on only the first 10% of
# each time series. This is a standard trick to avoid getting caught in a local
# minimum.
lr_strategy=(3e-3, 3e-4)
steps_strategy=(500, 5000)
length_strategy=(0.1, 1)


@eqx.filter_value_and_grad
def grad_loss(model, ti, yi):
    #input is the time points and the initial condition of timeseries in the batch
    y_pred = jax.vmap(model, in_axes=(None, 0))(ti, yi[:, 0])
    #L2 loss
    return jnp.mean((yi - y_pred) ** 2)


#%%



#%%
@eqx.filter_jit
def make_step(ti, yi, model, opt_state):
    loss, grads = grad_loss(model, ti, yi)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

for lr, iters, length in zip(lr_strategy, steps_strategy, length_strategy):
    optim = optax.adabelief(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    _ts = ts[: int(length_size * length)]
    _ys = ys[:, : int(length_size * length)]
    for iter, (yi,) in zip(
        range(iters), 
        dataloader((_ys,), batch_size, key=loader_key)
        ):
        start = time.time()
        loss, model, opt_state = make_step(_ts, yi, model, opt_state)
        end = time.time()
        if (iter % print_every) == 0 or iter == iters - 1:
            print(f"Iteration: {iter}, Loss: {loss}, Computation time: {end - start}")


#%%
if plot:
    plt.plot(ts, ys[0, :, 0], c="dodgerblue", label="Real")
    # plt.plot(ts, ys[0, :, 1], c="dodgerblue")
    # plt.plot(ts, ys[0, :, 2], c="dodgerblue")
    model_y = model(ts, ys[0, 0])
    plt.plot(ts, model_y[:, 0], c="crimson", label="Model")
    # plt.plot(ts, model_y[:, 1], c="crimson")
    # plt.plot(ts, model_y[:, 2], c="crimson")
    plt.legend()
    plt.tight_layout()
    plt.savefig("neural_ode_batch64_depth2_vdp_moreiters.png")
    plt.show()
# %%
