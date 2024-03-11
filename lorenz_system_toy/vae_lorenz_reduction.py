
import numpy as np
from scipy.integrate import solve_ivp
import jax.numpy as jnp
from jax import random, jit
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Relu, Softplus
import numpyro
from numpyro import optim
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

# Simulate Lorenz system
def simulate_lorenz_system(initial_state, t_span=[0., 50.], t_steps=10000):
    t_eval = np.linspace(t_span[0], t_span[1], t_steps)
    solution = solve_ivp(lorenz_system, t_span, initial_state, t_eval=t_eval)
    return solution.y

# Data preparation
initial_state = np.array([1., 1., 1.])
lorenz_data = simulate_lorenz_system(initial_state)

# Reshape and normalize data
lorenz_data = lorenz_data.reshape(3, -1).T  # Transpose to have shape (timepoints, variables)
lorenz_data = (lorenz_data - np.mean(lorenz_data, axis=0)) / np.std(lorenz_data, axis=0)

# VAE model components
def encoder(hidden_dim=100, z_dim=1):
    return stax.serial(
        Dense(hidden_dim), Relu,
        stax.FanOut(2),
        stax.parallel(
            Dense(z_dim),  # mean
            stax.serial(Dense(z_dim), Softplus)  # std
        )
    )

def decoder(hidden_dim=100, out_dim=3):
    return stax.serial(
        Dense(hidden_dim), Relu,
        Dense(out_dim)
    )

def model(batch):
    decode = numpyro.module("decoder", decoder(), input_shape=(-1, 1))
    with numpyro.plate("data", batch.shape[0]):
        z = numpyro.sample("z", dist.Normal(0, 1).to_event(1))
        reconstruction = decode(z)
        numpyro.sample("obs", dist.Normal(reconstruction, 0.1).to_event(1), obs=batch)

def guide(batch):
    encode = numpyro.module("encoder", encoder(), input_shape=(batch.shape[1],))
    z_loc, z_scale = encode(batch)
    numpyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

# Training
def train(data, num_epochs=1000, batch_size=128):
    optimizer = optim.Adam(step_size=1e-3)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    svi_state = svi.init(random.PRNGKey(0), data)
    
    for epoch in range(num_epochs):
        svi_state, loss = svi.update(svi_state, data)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")
    
    return svi.get_params(svi_state)

# Main
if __name__ == "__main__":
    params = train(lorenz_data)
    print("Training complete")
