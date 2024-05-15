import torch
import torchdiffeq as tde

def lorenz_system(t, y):
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    x, y, z = y.unbind()
    return torch.tensor([sigma * (y - x), x * (rho - z) - y, x * y - beta * z], device=y.device, dtype=torch.float32)

def generate_data(num_samples=1000, timesteps=100, initial_scale=10.0, max_time=2.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    initial_conditions = torch.randn(num_samples, 3, device=device) * initial_scale
    data = torch.zeros(timesteps, num_samples, 3, device=device)

    # Create CUDA streams
    streams = [torch.cuda.Stream(device=device) for _ in range(num_samples)]

    # Generate different timepoints for each initial condition
    start_times = torch.rand(num_samples, device=device) * max_time * 0.5
    time_vectors = [torch.linspace(start_times[i], start_times[i] + max_time * 0.5, timesteps, device=device) for i in range(num_samples)]

    for i, t in enumerate(time_vectors):
        with torch.cuda.stream(streams[i]):
            trajectory = tde.odeint(lorenz_system, initial_conditions[i], t)
            data[:, i, :].copy_(trajectory, non_blocking=True)

    # Wait for all streams to complete
    torch.cuda.synchronize(device=device)

    return data.permute(1, 0, 2)

# Example usage
data = generate_data()
