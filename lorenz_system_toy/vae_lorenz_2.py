import numpy as np
from scipy.integrate import solve_ivp
import jax.numpy as jnp
from jax import random
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Relu
import numpyro
from numpyro import optim
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.handlers import seed, module
# from numpyro.contrib.autoguide import AutoContinuousELBO

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

# Flatten and normalize trajectories
trajectories = trajectories.reshape(100, -1)  # Flatten trajectories
# Normalize per trajectory
trajectories = (trajectories - np.mean(trajectories, axis=1, keepdims=True)) / np.std(trajectories, axis=1, keepdims=True)

latent_dim = 10000  # High-dimensional latent space, though unconventional
learning_rate = 1e-4

# Neural network architectures
def encoder_net(input_dim, latent_dim):
    return stax.serial(
        Dense(512), Relu,
        Dense(256), Relu,
        stax.FanOut(2),
        stax.parallel(
            Dense(latent_dim),  # mean
            stax.serial(Dense(latent_dim), stax.Exp)  # std, ensuring it's positive
        )
    )


def decoder_net(output_dim):
    return stax.serial(
        Dense(256), Relu,
        Dense(512), Relu,
        Dense(output_dim)
    )


# Define the model
def model(batch):
    decoder = numpyro.module("decoder", decoder_net(3 * 10000), input_shape=(latent_dim,))
    with numpyro.plate("data", batch.shape[0]):
        # Sample the latent space
        z = numpyro.sample("z", dist.Normal(jnp.zeros(latent_dim), jnp.ones(latent_dim)).to_event(1))
        # Decode and sample the observed data
        reconstruction = decoder(z)
        numpyro.sample("obs", dist.Normal(reconstruction, 0.1).to_event(1), obs=batch)


# Define the guide (variational distribution)
def guide(batch):
    encoder = numpyro.module("encoder", encoder_net(batch.shape[-1], latent_dim), input_shape=(batch.shape[-1],))
    with numpyro.plate("data", batch.shape[0]):
        z_loc, z_scale = encoder(batch)
        # Sample the latent space
        numpyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))


# Initialize optimizer and SVI
optimizer = optim.Adam(step_size=learning_rate)
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

# Training function
def train(data, num_epochs=10, batch_size=5):
    rng_key = random.PRNGKey(0)
    svi_state = svi.init(rng_key, data[:batch_size])
    num_batches = data.shape[0] // batch_size

    for epoch in range(num_epochs):
        rng_key, _ = random.split(rng_key)
        epoch_loss = 0
        for i in range(num_batches):
            batch = data[i * batch_size:(i + 1) * batch_size]
            svi_state, loss = svi.update(svi_state, batch)
            epoch_loss += loss
        epoch_loss /= num_batches
        if epoch % 1 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss}")

    return svi.get_params(svi_state)


# Preprocess data: convert to JAX array and normalize
processed_data = jnp.array(trajectories, dtype=jnp.float32)

# Train the model
train_params = train(processed_data, num_epochs=100, batch_size=5)
