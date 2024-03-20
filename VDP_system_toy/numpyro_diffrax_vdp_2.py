# %%
import jax, diffrax
from jax import random
import jax.numpy as jnp
from jax.example_libraries import stax
from diffrax import diffeqsolve, ODETerm, Tsit5, PIDController, SaveAt


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
    ts = jnp.linspace(0, 10, 100)
    key = jr.split(key, dataset_size)
    ys = jax.vmap(lambda key: _get_data(ts, key=key))(key)
    return ts, ys

#%%
dataset_size = 512
ts, ys = get_data(dataset_size, key=jr.PRNGKey(1234))

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

#%%

def create_mlp_model(data_size, width_size, depth, activation=stax.Softplus):
    """Creates an MLP model with the specified architecture."""
    layers = []
    for _ in range(depth):
        layers.append(stax.Dense(width_size))
        layers.append(activation)
    layers.append(stax.Dense(data_size))  # Output layer to match data_size
    return stax.serial(*layers)

#%%
import numpyro
import numpyro.distributions as dist

def neural_ode_model(y0, ts):
    # Define the model size parameters and architecture again if needed
    data_size, width_size, depth = y0.shape[-1], 64, 2  # Example parameters

    # Define the Stax model (repeated here for clarity)
    mlp_model = create_mlp_model(data_size, width_size, depth)
    init_random_key = random.PRNGKey(0)  # Random key for initialization
    input_shape = (-1, data_size)  # Example input shape

    # Initialize the model to get the apply function (parameters are managed by NumPyro)
    _, params = mlp_model[0](init_random_key, input_shape)
    mlp_apply = mlp_model[1]

    # Wrap the neural network with numpyro.module
    # nn_params = numpyro.module("mlp", mlp_apply, (input_shape,), init_random_key)
    nn_params = numpyro.module("mlp", mlp_model, init_random_key)

    # Define the dynamics function using the wrapped parameters
    def dynamics_func(t, y, args):
        return mlp_apply(nn_params, y)

    # Solve the ODE with Diffrax, using the dynamics defined by the neural network
    solution = diffeqsolve(
        ODETerm(dynamics_func),
        Tsit5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=y0,
        stepsize_controller=PIDController(rtol=1e-3, atol=1e-6),
        saveat=SaveAt(ts=ts),
    )

    return solution.ys

# %%
import jax.numpy as jnp
from jax import random
import jax.random as jr
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Relu, Softplus
import numpyro
from numpyro import optim
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from jax import jit
from numpyro.optim import Adam
from jax import lax
import matplotlib.pyplot as plt
import os
from jax.random import PRNGKey

# Define the encoder and decoder architectures
def encoder(hidden_dim, z_dim):
    return stax.serial(
        Dense(hidden_dim),
        Softplus,
        stax.FanOut(2),
        stax.parallel(
            Dense(z_dim),
            stax.serial(Dense(z_dim), Softplus),
        ),
    )

def decoder(hidden_dim, out_dim):
    return stax.serial(
        Dense(hidden_dim),
        Softplus,
        Dense(out_dim),
        stax.Sigmoid,  # You may consider removing this if your data isn't in [0,1]
    )


#%%
def model(batch, ts, hidden_dim=2, z_dim=1, output_dim=2, width_size=64, depth=2, key=jr.PRNGKey(0)):
    batch_dim = batch.shape[0]

    decode = numpyro.module("decoder", decoder(hidden_dim, output_dim), (batch_dim, z_dim))

    # Prior distribution for latent variables
    with numpyro.plate("data", batch_dim):
        z = numpyro.sample("z", dist.Normal(0, 1).expand((z_dim,)).to_event(1))
        z_hat = neural_ode_model(z[0,:], ts)
        # Direct decoding branch
        predicted_state = decode(z_hat)
        numpyro.sample("obs_direct", dist.Normal(predicted_state, 1).to_event(1), obs=batch)


def guide(batch, ts, hidden_dim=2, z_dim=1, output_dim=3, width_size=64, depth=2, key=jr.PRNGKey(0)):
    # batch = jnp.reshape(batch, (batch.shape[0] * batch.shape[1], -1))  # Reshape to process each time point individually
    batch_dim = batch.shape[0]
    encode = numpyro.module("encoder", encoder(hidden_dim, z_dim), (batch_dim, output_dim))

    z_loc, z_std = encode(batch)

    #I want the batch to hold basically the timeseries of multiple samples but simple for now just 1 timeseries at a time
    with numpyro.plate("data", batch_dim):
        return numpyro.sample("z", dist.Normal(z_loc, jnp.exp(z_std)).to_event(1))

# %%
# Configuration Parameters
input_dim = output_dim = 1#ys.shape[1] #3
hidden_dim = 2
z_dim = 1
learning_rate = 3e-3
# batch0_size = 64
batch_size = len(ts)# * batch0_size
num_epochs = 10
RESULTS_DIR = "./vae_results"

# Make sure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Initialize Neural Networks, Optimizer, and SVI
encoder_nn = encoder(hidden_dim, z_dim)
decoder_nn = decoder(hidden_dim, output_dim)
adam = optim.Adam(learning_rate)
svi = SVI(model, guide, adam, Trace_ELBO(), hidden_dim=hidden_dim, z_dim=z_dim, output_dim=output_dim, width_size=64, depth=2, key=jr.PRNGKey(0))

# PRNG Keys
rng_key = PRNGKey(0)

#%%
import time
from numpyro.diagnostics import hpdi
import numpyro
numpyro.set_platform("cpu")  # Use GPU if available: numpyro.set_platform("gpu")

# Split the RNG key for initializing the SVI state
rng_key, rng_key_init = random.split(rng_key)

# svi_result = svi.run(random.PRNGKey(0), 2, train_batches[0])

# Initialize the SVI state using a dummy batch
# Adjust the shape of the dummy batch (jnp.ones(...)) as necessary for your model's input
svi_state = svi.init(rng_key_init, jnp.ones((batch_size, input_dim)), ts)



#%%
# for epoch in range(num_epochs):
train_loss = 0.0
# Iterate over the training data
# for i, batch in enumerate(train_batches):
# Prepare the batch - ensure it's in the correct format
epoch = 0
for i, (batch,) in zip(range(100),dataloader((ys,), batch_size=1, key=key)):
    # Update the SVI state and compute loss for this batch
    svi_state, loss = svi.update(svi_state, batch)
    print(f"Epoch {epoch+1}, Batch {i+1}: Loss = {loss}")

# %%
