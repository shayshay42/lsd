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
from scipy.integrate import solve_ivp
import torch
from torch.utils.data import DataLoader, TensorDataset

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
trajectories = trajectories.reshape(1000, -1)  # Flatten trajectories
# Normalize per trajectory
trajectories = (trajectories - np.mean(trajectories, axis=1, keepdims=True)) / np.std(trajectories, axis=1, keepdims=True)

# Create train and test dataloaders
batch_size = 100

# First 80% of trajectories are used for training
train_dataset = TensorDataset(torch.Tensor(trajectories[:800]))
test_dataset = TensorDataset(torch.Tensor(trajectories[800:]))

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trajectories.shape, device


# %%
# from .autonotebook import tqdm as notebook_tqdm
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
# from numpyro.handlers import seed, module


def encoder(hidden_dim, z_dim):
    return stax.serial(
        stax.Dense(hidden_dim, W_init=stax.randn()),
        stax.Softplus,
        stax.FanOut(2),
        stax.parallel(
            stax.Dense(z_dim, W_init=stax.randn()),
            stax.serial(stax.Dense(z_dim, W_init=stax.randn()), stax.Exp),
        ),
    )

def decoder(hidden_dim, out_dim):
    return stax.serial(
        stax.Dense(hidden_dim, W_init=stax.randn()),
        stax.Softplus,
        stax.Dense(out_dim, W_init=stax.randn()),
        stax.Sigmoid,
    )


# def model(batch, hidden_dim=5000, z_dim=10000):
#     batch = jnp.reshape(batch, (batch.shape[0], -1))
#     batch_dim, out_dim = jnp.shape(batch)
#     decode = numpyro.module("decoder", decoder(hidden_dim, out_dim), (batch_dim, z_dim))
#     with numpyro.plate("batch", batch_dim):
#         z = numpyro.sample("z", dist.Normal(0, 1).expand([z_dim]).to_event(1))
#         img_loc = decode(z)
#         return numpyro.sample("obs", dist.Bernoulli(img_loc).to_event(1), obs=batch)

# def model(batch, hidden_dim=5000, z_dim=10000, out_dim=30000):
#     batch = jnp.reshape(batch, (batch.shape[0], -1))  # Flatten the batch
#     batch_dim = batch.shape[0]
#     decode = numpyro.module("decoder", decoder(hidden_dim, out_dim), (batch_dim, out_dim))
#     with numpyro.plate("batch", batch_dim):
#         z = numpyro.sample("z", dist.Normal(0, 1).expand((z_dim,)).to_event(1))
#         predicted_states = decode(z)
#         numpyro.sample("obs", dist.Normal(predicted_states, 1).to_event(1), obs=batch)


# def guide(batch, hidden_dim=5000, z_dim=10000):
#     batch = jnp.reshape(batch, (batch.shape[0], -1))
#     batch_dim, out_dim = jnp.shape(batch)
#     encode = numpyro.module("encoder", encoder(hidden_dim, z_dim), (batch_dim, out_dim))
#     z_loc, z_std = encode(batch)
#     with numpyro.plate("batch", batch_dim):
#         return numpyro.sample("z", dist.Normal(z_loc, z_std).to_event(1))
    
# def guide(batch, hidden_dim=5000, z_dim=10000):
#     batch = jnp.reshape(batch, (batch.shape[0], -1))  # Flatten the batch
#     batch_dim = batch.shape[0]
#     encode = numpyro.module("encoder", encoder(hidden_dim, z_dim), (batch_dim, z_dim))
#     z_loc, z_std = encode(batch)
#     with numpyro.plate("batch", batch_dim):
#         numpyro.sample("z", dist.Normal(z_loc, jnp.exp(z_std)).to_event(1))
def model(batch, hidden_dim=5000, z_dim=10000, out_dim=30000):
    batch = jnp.reshape(batch, (batch.shape[0], -1))  # Flatten the batch
    batch_dim = batch.shape[0]
    # Initialize the decoder, specifying that its input will have dimensions (batch_dim, z_dim)
    decode = numpyro.module("decoder", decoder(hidden_dim, out_dim), (batch_dim, z_dim))
    
    with numpyro.plate("batch", batch_dim):
        # Sample z from the prior
        z = numpyro.sample("z", dist.Normal(0, 1).expand((z_dim,)).to_event(1))
        # Decode z to reconstruct the data
        predicted_states = decode(z)
        # Sample observations from a Normal distribution centered around the predictions
        numpyro.sample("obs", dist.Normal(predicted_states, 1).to_event(1), obs=batch)

def guide(batch, hidden_dim=5000, z_dim=10000):
    batch = jnp.reshape(batch, (batch.shape[0], -1))  # Flatten the batch for processing
    batch_dim = batch.shape[0]
    # Initialize the encoder; it takes the original data and outputs parameters for the latent space
    # The encoder outputs two vectors of size z_dim: mean (z_loc) and log variance (z_std)
    encode = numpyro.module("encoder", encoder(hidden_dim, z_dim), (batch_dim, z_dim))
    
    # Encode the input data to get the parameters of the latent space distribution
    z_loc, z_std = encode(batch)
    
    # Sample z from the approximate posterior defined by the encoder's outputs
    with numpyro.plate("batch", batch_dim):
        numpyro.sample("z", dist.Normal(z_loc, jnp.exp(z_std)).to_event(1))

# batch = jnp.reshape(batch, (batch.shape[0], -1))  # Flatten the batch

# %%
import jax.numpy as jnp
from jax import random, jit
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam
from numpyro import optim
from jax import lax
import matplotlib.pyplot as plt
import os
from jax.random import PRNGKey

# Configuration Parameters
input_dim = output_dim = trajectories.shape[1] #30000
hidden_dim = 5000
z_dim = 10000
learning_rate = 1e-3
batch_size = 100
num_epochs = 10
RESULTS_DIR = "./vae_results"

# Make sure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Initialize Neural Networks, Optimizer, and SVI
encoder_nn = encoder(hidden_dim, z_dim)
decoder_nn = decoder(hidden_dim, output_dim)
adam = optim.Adam(learning_rate)
svi = SVI(model, guide, adam, Trace_ELBO(), hidden_dim=hidden_dim, z_dim=z_dim)

# PRNG Keys
rng_key = PRNGKey(0)

# %%
# Import additional required libraries
import time
from numpyro.diagnostics import hpdi
import numpyro
numpyro.set_platform("cpu")  # Use GPU if available: numpyro.set_platform("gpu")

# Evaluation Function
def eval_test(svi, svi_state, rng_key, test_loader):
    test_loss = 0.0
    for i, (batch,) in enumerate(test_loader):
        rng_key, _ = random.split(rng_key)
        batch = np.array(batch)  # Ensure batch is in the correct format
        loss = svi.evaluate(svi_state, batch)
        test_loss += loss
    test_loss /= len(test_loader)
    return test_loss

def epoch_train(svi, rng_key, train_loader, svi_state):
    train_loss = 0.0
    for i, (batch,) in enumerate(train_loader):
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
run_training(svi, rng_key, train_loader, test_loader, num_epochs=num_epochs)


# %%
import matplotlib.pyplot as plt
import numpy as np

def reconstruct(svi, svi_state, batch):
    """
    Reconstructs the given batch of data using the trained VAE.
    This function needs to extract the decoder part from the VAE model
    and apply it to the latent variables sampled from the posterior.
    """
    # Extract the parameters for the decoder
    params = svi.get_params(svi_state)
    z_loc, z_std = encoder_nn(batch)  # You need to define how to get these from your model
    z = dist.Normal(z_loc, z_std).sample()
    reconstructed_batch = decoder_nn(z)
    return reconstructed_batch

def plot_lorenz_original_vs_reconstructed(svi, svi_state, test_loader):
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    for i, (batch,) in enumerate(test_loader):
        if i >= 5: break
        original_data = np.array(batch)
        reconstructed_data = reconstruct(svi, svi_state, original_data)
        
        # Assuming the data shape is (time_steps, 3) for each sample
        t = np.arange(original_data.shape[1])
        
        # Original Data Plot
        axs[0, i].plot(t, original_data[0, :, 0], 'r', label='X')
        axs[0, i].plot(t, original_data[0, :, 1], 'g', label='Y')
        axs[0, i].plot(t, original_data[0, :, 2], 'b', label='Z')
        axs[0, i].set_title('Original')
        if i == 0: axs[0, i].legend()
        
        # Reconstructed Data Plot
        axs[1, i].plot(t, reconstructed_data[0, :, 0], 'r', label='X')
        axs[1, i].plot(t, reconstructed_data[0, :, 1], 'g', label='Y')
        axs[1, i].plot(t, reconstructed_data[0, :, 2], 'b', label='Z')
        axs[1, i].set_title('Reconstructed')
        if i == 0: axs[1, i].legend()

    plt.tight_layout()
    plt.show()

# Generate and plot reconstructions from the test dataset
plot_lorenz_original_vs_reconstructed(svi, svi_state, test_loader)
