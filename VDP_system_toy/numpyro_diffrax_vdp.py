#%%
import time

import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax  # https://github.com/deepmind/optax

class Func(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=data_size,
            out_size=data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            key=key,
        )

    def __call__(self, t, y, args):
        return self.mlp(y)
    

class NeuralODE(eqx.Module):
    func: Func

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.func = Func(data_size, width_size, depth, key=key)

    def __call__(self, ts, y0):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return solution.ys
    

# %%

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
#plot the data
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

# loader_key = jr.PRNGKey(1234)
# _ys = dataloader((ys,), batch_size=32, key=loader_key)

# %%
import jax.numpy as jnp
from jax import random
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

#uniform distribution?????????????????????????????????????????????????????????????????????????

# Adjust the model and guide functions for handling time points individually
# def model(batch, hidden_dim=2, z_dim=1, output_dim=3):
#     # batch = jnp.reshape(batch, (batch.shape[0] * batch.shape[1], -1))  # Reshape to process each time point individually
#     batch_dim = batch.shape[0]
#     decode = numpyro.module("decoder", decoder(hidden_dim, output_dim), (batch_dim, z_dim))
    
#     with numpyro.plate("data", batch_dim):
#         z = numpyro.sample("z", dist.Normal(0, 1).expand((z_dim,)).to_event(1))
#         predicted_state = decode(z)
#         return numpyro.sample("obs", dist.Normal(predicted_state, 1).to_event(1), obs=batch)

def model(batch, ts, hidden_dim=2, z_dim=1, output_dim=3, width_size=64, depth=2, key=jr.PRNGKey(0)):
    batch_dim = batch.shape[0]
    # Assuming the time dimension is the second axis
    series_length = batch.shape[1] 

    decode = numpyro.module("decoder", decoder(hidden_dim, output_dim), (batch_dim, z_dim))

    neural_ode = numpyro.module("neuralode", NeuralODE(z_dim, width_size, depth, key=key))

    # Prior distribution for latent variables
    with numpyro.plate("data", batch_dim):
        z = numpyro.sample("z", dist.Normal(0, 1).expand((z_dim,)).to_event(1))

        # Direct decoding branch
        predicted_state = decode(z)
        numpyro.sample("obs_direct", dist.Normal(predicted_state, 1).to_event(1), obs=batch)

        # Neural ODE branch
        #if batch_id == 0:
        initial_condition = z[0,:]
        ode_solution = neural_ode(ts, initial_condition)
        decoded_ode_solution = jax.vmap(decode)(ode_solution)#.reshape(batch_dim, series_length, output_dim)

        # Sample the observed data from the distribution parameterized by the decoded ODE solution
        numpyro.sample("obs_ode", dist.Normal(decoded_ode_solution, 1).to_event(2), obs=batch)


def guide(batch, ts, hidden_dim=2, z_dim=1, output_dim=3, width_size=64, depth=2, key=jr.PRNGKey(0)):
    # batch = jnp.reshape(batch, (batch.shape[0] * batch.shape[1], -1))  # Reshape to process each time point individually
    batch_dim = batch.shape[0]
    encode = numpyro.module("encoder", encoder(hidden_dim, z_dim), (batch_dim, output_dim))
    
    z_loc, z_std = encode(batch)

    #I want the batch to hold basically the timeseries of multiple samples but simple for now just 1 timeseries at a time
    with numpyro.plate("data", batch_dim):
        return numpyro.sample("z", dist.Normal(z_loc, jnp.exp(z_std)).to_event(1))

#%%




#%%
# print(encoder(hidden_dim, z_dim))        

# %%

# Configuration Parameters
input_dim = output_dim = ys.shape[1] #3
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
svi_state = svi.init(rng_key_init, jnp.ones((batch_size, input_dim)), ts)

# Begin training loop
# num_epochs = 2
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
batch_size=128

width_size=64
depth=3
seed=5678
plot=True
print_every=100

key = jr.PRNGKey(seed)
data_key, model_key, loader_key = jr.split(key, 3)

# ts, ys = t_eval, train_batches
dataset_size, length_size, data_size = ys.shape

model = NeuralODE(data_size, width_size, depth, key=model_key)

lr_strategy=(3e-3, 3e-4)
steps_strategy=(1000, 5000)
length_strategy=(0.1, 1)

@eqx.filter_value_and_grad
def grad_loss(model, ti, yi):
    batch_size, timepoints, _ = yi.shape

    # Predict the integrated states using the model for each initial condition
    # Note: This assumes model can be applied to a single initial condition and produce a trajectory
    y_pred = jax.vmap(lambda y0: model(ti, y0), in_axes=(0))(yi[:, 0, :])

    # Compute true derivatives for every point in the trajectory
    # Flatten yi for vmap, then reshape to original
    yi_flat = yi.reshape(-1, yi.shape[-1])
    y_dot_true_flat = jax.vmap(f, in_axes=(None, 0, None))(None, yi_flat, None)
    y_dot_true = y_dot_true_flat.reshape(batch_size, timepoints, -1)

    # Compute predicted derivatives for every point in the trajectory
    # Similar flattening approach for y_dot_pred
    y_dot_pred_flat = jax.vmap(model.func, in_axes=(None, 0, None))(None, yi_flat, None)
    y_dot_pred = y_dot_pred_flat.reshape(batch_size, timepoints, -1)

    # Calculate the mean squared error (MSE) for the predictions
    mse_loss = jnp.mean((yi - y_pred) ** 2)
    
    # Calculate the MSE for the derivatives
    physics_loss = jnp.mean((y_dot_true - y_dot_pred) ** 2)

    return mse_loss + physics_loss

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
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.experimental import stax
from jax.experimental.stax import Dense, Softplus
import equinox as eqx
import diffrax


def combined_model(batch, ts, hidden_dim=2, z_dim=1, output_dim=3, data_size=3, width_size=64, depth=2, key=jr.PRNGKey(0)):
    batch_dim, timepoints, _ = batch.shape
    decode = numpyro.module("decoder", decoder(hidden_dim, output_dim), (batch_dim, z_dim))

    # Neural ODE part
    neural_ode = NeuralODE(data_size, width_size, depth, key=key)

    with numpyro.plate("data", batch_dim):
        z = numpyro.sample("z", dist.Normal(z_loc, jnp.exp(z_std)).to_event(1))
        ode_solution = neural_ode(ts, z)  # Solve the ODE using z as initial conditions

        # Decode the ODE solution (trajectory)
        predicted_trajectory = jax.vmap(decode)(ode_solution.reshape(-1, z_dim)).reshape(batch_dim, timepoints, output_dim)
        numpyro.sample("obs", dist.Normal(predicted_trajectory, 1).to_event(2), obs=batch)


def combined_guide(batch, ts, hidden_dim=2, z_dim=1, output_dim=3, data_size=3, width_size=64, depth=2, key=jr.PRNGKey(0)):
    batch_dim, timepoints, _ = batch.shape
    # Note: For the guide, you typically don't need to pass the output_dim of the decoder
    # but rather the input dimension that the encoder expects.
    encode = numpyro.module("encoder", encoder(hidden_dim, z_dim), (batch_dim, data_size))
    
    # Reshape the batch to pass individual time points to the encoder
    batch_reshaped = batch.reshape(-1, data_size)  # Assuming data_size matches the flattened input dimension
    z_loc, z_std = encode(batch_reshaped)
    
    # Reshape z_loc and z_std back to the original batch structure and take means
    # These operations are assuming you want a single set of parameters per batch item, across timepoints
    z_loc = z_loc.reshape(batch_dim, timepoints, -1).mean(axis=1)  # Mean location across timepoints
    z_std = z_std.reshape(batch_dim, timepoints, -1).mean(axis=1)  # Mean scale (std. dev.) across timepoints

    with numpyro.plate("data", batch_dim):
        # Sample the latent variable z from the variational distribution
        # Note: You might consider parameterizing the scale (std. dev.) with softplus to ensure it's positive
        z_scale_positive = jnp.exp(z_std)
        numpyro.sample("z", dist.Normal(z_loc, z_scale_positive).to_event(1))

