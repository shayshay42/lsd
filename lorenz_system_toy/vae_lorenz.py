import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import random
from jax.example_libraries import stax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro import optim

import jax.numpy as jnp
from jax import random
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Relu
import numpyro
from numpyro import optim
from numpyro.contrib.module import random_flax_module
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO

# Lorenz system parameters
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# Lorenz system differential equations
def lorenz_system(t, state):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Time span for the simulation
t_span = (0, 50)
t_eval = np.linspace(*t_span, 10000)

# Initial conditions
initial_conditions = np.random.rand(100, 3) * 20 - 10  # Random initial conditions

# Simulate the Lorenz system for 100 different initial conditions
trajectories = np.array([solve_ivp(lorenz_system, t_span, ic, t_eval=t_eval).y for ic in initial_conditions])


# Correcting the model to handle batch dimensions properly
def model(batch):
    batch_dim, _, _ = batch.shape  # Assuming batch shape is [batch_size, 3, 10000]
    batch = batch.reshape(batch_dim, -1)  # Flatten the input

    # numpyro.plate for handling batch dimensions
    with numpyro.plate("data", batch_dim):
        # Sample the latent space
        z = numpyro.sample("z", dist.Normal(0, 1).expand([latent_dim]).to_event(1))

        # Define the decoder network
        decoder_net = numpyro.module("decoder", stax.serial(
            Dense(512), Relu,
            Dense(1024), Relu,
            Dense(batch.shape[-1]),  # Match the flattened input dimension
        ), input_shape=(-1, latent_dim))

        # Decode the latent variable
        reconstructed = decoder_net(z)
        # Likelihood of the observations
        numpyro.sample("x", dist.Normal(reconstructed, 0.1).to_event(1), obs=batch)

# Correcting the guide to handle batch dimensions properly
def guide(batch):
    batch_dim, _, _ = batch.shape  # Assuming batch shape is [batch_size, 3, 10000]
    batch = batch.reshape(batch_dim, -1)  # Flatten the input

    # numpyro.plate for handling batch dimensions
    with numpyro.plate("data", batch_dim):
        # Define the encoder network
        encoder_net = numpyro.module("encoder", stax.serial(
            Dense(1024), Relu,
            Dense(512), Relu,
            stax.FanOut(2),
            stax.parallel(
                Dense(latent_dim),  # For the mean of the latent space
                stax.serial(Dense(latent_dim), stax.Exp)  # For the std of the latent space, ensuring positive values
            )
        ), input_shape=(-1, 3*10000))

        # Encoder forward pass
        z_loc, z_scale = encoder_net(batch)

        # Sample the latent variable
        numpyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

latent_dim = 10000  # Size of the latent space
learning_rate = 1e-3

# Initialize the optimizer
optimizer = numpyro.optim.Adam(step_size=learning_rate)

# Initialize the model and guide
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

# Your training loop and data preparation remain the same

# Dummy data for illustration
rng_key = random.PRNGKey(0)
num_samples = 100  # Number of samples (trajectories)
dummy_data = trajectories

# Training loop
num_epochs = 1000
batch_size = 5

def train(dummy_data):
    rng_key = random.PRNGKey(0)
    svi_state = svi.init(rng_key, dummy_data)

    for epoch in range(num_epochs):
        rng_key, _ = random.split(rng_key)
        svi_state, loss = svi.update(svi_state, dummy_data)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    return svi.get_params(svi_state)

# Train the model
train_params = train(dummy_data)