#%%
import jax, diffrax
from jax import random
import jax.numpy as jnp
import jax.random as jr
from jax.example_libraries import stax
from diffrax import diffeqsolve, ODETerm, Tsit5, PIDController, SaveAt
import optax
from jax import value_and_grad, jit
import numpyro
from numpyro import optim

#seed and keys
seed = 1234
key = random.PRNGKey(seed)
model_key, data_key, loader_key = random.split(key, 3)


def f(t, y, args):
    mu = 3.0
    x, v = y
    dxdt = v
    dvdt = mu * (1 - x**2) * v - x
    return jnp.stack([dxdt, dvdt], axis=-1)

def _get_data(ts, *, key):
    y0 = jr.uniform(key, (2,), minval=-2, maxval=2)

    solver = diffrax.Tsit5()
    dt0 = 0.1
    saveat = diffrax.SaveAt(ts=ts)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(f), solver, ts[0], ts[-1], dt0, y0, saveat=saveat
    )
    ys = sol.ys
    return ys


def get_data(dataset_size, *, key):
    ts = jnp.linspace(0, 10, 200)
    key = jr.split(key, dataset_size)
    ys = jax.vmap(lambda key: _get_data(ts, key=key))(key)
    return ts, ys

#%%
dataset_size = 512
ts, ys = get_data(dataset_size, key=data_key)

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


def create_mlp_model(data_size, width_size, depth, activation=stax.Softplus):
    """Creates an MLP model with the specified architecture."""
    layers = []
    for _ in range(depth):
        layers.append(stax.Dense(width_size))
        layers.append(activation)
    layers.append(stax.Dense(data_size))  # Output layer to match data_size
    return stax.serial(*layers)

#%%
state_size, width, depth = ys.shape[-1], 64, 2  # Example parameters

# Define the Stax model (repeated here for clarity)
mlp_model = create_mlp_model(state_size, width, depth)
input_shape = (-1, state_size)  # Example input shape

# Initialize the model to get the apply function (parameters are managed by NumPyro)
_, params = mlp_model[0](model_key, input_shape)
mlp_apply = mlp_model[1]

# Solve the ODE with Diffrax, using the dynamics defined by the neural network
def ode_solve(y0, ts, params):
    solution = diffeqsolve(
        ODETerm(lambda t, y, args: mlp_apply(params, y)),
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
def mse_loss(params, y0, ts, targets):
    """Computes the mean squared error loss."""
    preds = ode_solve(y0, ts, params)  # Use the updated ode_solve that uses the MLP
    #reshape to 64 100 2
    preds = jnp.transpose(preds, axes=(1, 0, 2))
    return jnp.mean((preds - targets) ** 2)

@jit
def update(params, opt_state, ts, targets):
    """Performs a single optimization step."""
    loss, grads = value_and_grad(mse_loss)(params, targets[:,0,:], ts, targets)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss

# Assuming neural_ode_model and other setup is defined correctly
# Setup optimizer
optimizer = optax.adabelief(learning_rate=3e-3)
opt_state = optimizer.init(params)

#%%
# Training loop

batch_size = 64

iters = 1000
num_epochs = 10

for epoch in range(num_epochs):
    for iteration, (yi,) in zip(range(iters), dataloader((ys,), batch_size, key=loader_key)):
        params, opt_state, loss = update(params, opt_state, ts, yi)
        print(f"Epoch {epoch}, Iteration {iteration}, Loss: {loss}")

#%%
# plot the results
import matplotlib.pyplot as plt
plt.plot(ts, ys[0, :, 0], label="True x")
plt.plot(ts, ys[0, :, 1], label="True v")
plt.plot(ts, ode_solve(ys[0, 0, :], ts, params)[:, 0], label="Predicted x")
plt.plot(ts, ode_solve(ys[0, 0, :], ts, params)[:, 1], label="Predicted v")
plt.legend()
plt.show()
# %%
fig, axs = plt.subplots(2, 1, figsize=(10, 8))  # Create two panels

# Solve the ODE for the MLP model's predictions
predicted_ys = ode_solve(ys[0, 0, :], ts, params)

# Plot true and predicted x on the first panel
axs[0].plot(ts, ys[0, :, 0], color='blue', label="True x")
axs[0].plot(ts, predicted_ys[:, 0], '--', color='blue', label="Predicted x")  # Dashed line for prediction

# Plot true and predicted v on the second panel
axs[1].plot(ts, ys[0, :, 1], color='orange', label="True v")
axs[1].plot(ts, predicted_ys[:, 1], '--', color='orange', label="Predicted v")  # Dashed line for prediction

# Set titles and labels
axs[0].set_title('Dynamics of x')
axs[1].set_title('Dynamics of v')
for ax in axs:
    ax.set_xlabel('Time')
    ax.legend()

plt.tight_layout()
plt.show()

# %%
