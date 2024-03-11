import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

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

# `trajectories` now contains the simulation results for 100 different initial conditions
