# %%
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
# create a transofrm to apply to each datapoint
transform = transforms.Compose([transforms.ToTensor()])

# %%
import numpy as np
import jax.numpy as jnp
from scipy.integrate import solve_ivp

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
initial_conditions = np.random.rand(1000, 3) * 20 - 10  # Random initial conditions

# Simulate the Lorenz system for 100 different initial conditions
trajectories = np.array([solve_ivp(lorenz_system, t_span, ic, t_eval=t_eval).y for ic in initial_conditions])

# Flatten and normalize trajectories
# trajectories = trajectories.reshape(-1, 3)  # Flatten trajectories
# Normalize per trajectory
# trajectories = (trajectories - np.mean(trajectories, axis=1, keepdims=True)) / np.std(trajectories, axis=1, keepdims=True)

# Create train and test dataloaders
batch_size = trajectories.shape[-1]  # Use a batch size equal to the number of timepoints

fraction = int(trajectories.shape[0] * 0.8)


# Convert to JAX arrays for compatibility
trajectories_jax = jnp.array(trajectories)

# Determine the split index
split_idx = int(trajectories_jax.shape[0] * 0.8)

# Split the dataset
train_batches = train_data = jnp.transpose(trajectories_jax[:split_idx], axes=(0, 2, 1))
test_batches = test_data = jnp.transpose(trajectories_jax[split_idx:], axes=(0, 2, 1))

# def batch_data(data, batch_size):
#     """Yield batches of data."""
#     num_batches = np.ceil(data.shape[0] / batch_size).astype(int)
#     for i in range(num_batches):
#         start = i * batch_size
#         end = start + batch_size
#         yield data[start:end]

# Prepare batches for the training data
# train_batches = list(batch_data(train_data, batch_size))
# Prepare batches for the testing data
# test_batches = list(batch_data(test_data, batch_size))

# Example usage
print(f"Number of training batches: {len(train_batches)}")
print(f"Number of testing batches: {len(test_batches)}")



# %%
import jax.numpy as jnp
from jax import random
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Relu, Softplus
import numpyro
from numpyro import optim
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO

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

#uniform distribution

# Adjust the model and guide functions for handling time points individually
def model(batch, hidden_dim=2, z_dim=1, output_dim=3):
    # batch = jnp.reshape(batch, (batch.shape[0] * batch.shape[1], -1))  # Reshape to process each time point individually
    batch_dim = batch.shape[0]
    decode = numpyro.module("decoder", decoder(hidden_dim, output_dim), (batch_dim, z_dim))
    
    with numpyro.plate("data", batch_dim):
        z = numpyro.sample("z", dist.Normal(0, 1).expand((z_dim,)).to_event(1))
        predicted_state = decode(z)
        numpyro.sample("obs", dist.Normal(predicted_state, 1).to_event(1), obs=batch)

def guide(batch, hidden_dim=2, z_dim=1, output_dim=3):
    # batch = jnp.reshape(batch, (batch.shape[0] * batch.shape[1], -1))  # Reshape to process each time point individually
    batch_dim = batch.shape[0]
    encode = numpyro.module("encoder", encoder(hidden_dim, z_dim), (batch_dim, output_dim))
    
    z_loc, z_std = encode(batch)
    with numpyro.plate("data", batch_dim):
        numpyro.sample("z", dist.Normal(z_loc, jnp.exp(z_std)).to_event(1))

#%%
# print(encoder(hidden_dim, z_dim))        

# %%
from jax import random, jit
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam
from jax import lax
import matplotlib.pyplot as plt
import os
from jax.random import PRNGKey

# Configuration Parameters
input_dim = output_dim = trajectories.shape[1] #3
hidden_dim = 2
z_dim = 1
learning_rate = 1e-3
batch_size = 10000
num_epochs = 1
RESULTS_DIR = "./vae_results"

# Make sure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Initialize Neural Networks, Optimizer, and SVI
encoder_nn = encoder(hidden_dim, z_dim)
decoder_nn = decoder(hidden_dim, output_dim)
adam = optim.Adam(learning_rate)
svi = SVI(model, guide, adam, Trace_ELBO(), hidden_dim=hidden_dim, z_dim=z_dim, output_dim=output_dim)

# PRNG Keys
rng_key = PRNGKey(0)


#%%
# Import additional required libraries
import time
from numpyro.diagnostics import hpdi
import numpyro
numpyro.set_platform("cpu")  # Use GPU if available: numpyro.set_platform("gpu")

# Split the RNG key for initializing the SVI state
rng_key, rng_key_init = random.split(rng_key)

# svi_result = svi.run(random.PRNGKey(0), 2, train_batches[0])

# Initialize the SVI state using a dummy batch
# Adjust the shape of the dummy batch (jnp.ones(...)) as necessary for your model's input
svi_state = svi.init(rng_key_init, jnp.ones((batch_size, input_dim)))

# Begin training loop
# num_epochs = 2
# for epoch in range(num_epochs):
train_loss = 0.0
# Iterate over the training data
# for i, batch in enumerate(train_batches):
# Prepare the batch - ensure it's in the correct format
epoch = 0
for i, batch in enumerate(train_batches):
    batch = np.array(train_batches[0])

    # Update the SVI state and compute loss for this batch
    svi_state, loss = svi.update(svi_state, batch)
    print(f"Epoch {epoch+1}, Batch {i+1}: Loss = {loss}")

# %%
# Import additional required libraries
import time
from numpyro.diagnostics import hpdi
import numpyro
numpyro.set_platform("cpu")  # Use GPU if available: numpyro.set_platform("gpu")

# Evaluation Function
def eval_test(svi, svi_state, rng_key, test_loader):
    test_loss = 0.0
    for i, batch in enumerate(test_loader):
        rng_key, _ = random.split(rng_key)
        batch = np.array(batch)  # Ensure batch is in the correct format
        loss = svi.evaluate(svi_state, batch)
        test_loss += loss
    test_loss /= len(test_loader)
    return test_loss

def epoch_train(svi, rng_key, train_loader, svi_state):
    train_loss = 0.0
    for i, batch in enumerate(train_loader):
        rng_key, _ = random.split(rng_key)
        batch = np.array(batch)  # Convert batch to the correct format if necessary
        svi_state, loss = svi.update(svi_state, batch)
        train_loss += loss
    train_loss /= len(train_loader)
    return svi_state, train_loss

def run_training(svi, rng_key, train_loader, test_loader, num_epochs=10):
    # Initialize svi_state here
    rng_key, rng_key_init = random.split(rng_key)
    svi_state = svi.init(rng_key_init, jnp.ones((batch_size, input_dim)))  # Adjust the init batch as necessary
    
    for epoch in range(num_epochs):
        rng_key, rng_key_train, rng_key_eval = random.split(rng_key, 3)
        t_start = time.time()
        
        # Pass svi_state to epoch_train and update it
        svi_state, train_loss = epoch_train(svi, rng_key_train, train_loader, svi_state)
        
        # Evaluation phase should also correctly handle svi_state
        test_loss = eval_test(svi, svi_state, rng_key_eval, test_loader)  # Assuming eval_test is similarly updated to accept svi_state

        print(f"Epoch {epoch}: Train loss = {train_loss}, Test loss = {test_loss} ({time.time() - t_start:.2f}s)")

# Execute the training
run_training(svi, rng_key, train_batches, test_batches, num_epochs=5)

#%%
params = svi.get_params(svi_state)
batch = np.array(train_batches[0])  # Example: using the first batch from your training data
z_mean, z_var = encoder_nn[1](
    params["encoder$params"], batch)

dist_pertime = dist.Normal(z_mean, jnp.exp(z_var / 2))
# %%
a = dist_pertime.sample(jr.PRNGKey(1))