import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint
import matplotlib.pyplot as plt

# Define the Lorenz system
def lorenz_system(t, y, sigma=10., beta=8./3, rho=28.):
    x, y, z = y[..., 0], y[..., 1], y[..., 2]
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return torch.stack([dx, dy, dz], dim=-1)  # Stack along the last dimension

# Generate dataset with different initial conditions
def generate_data(num_samples=1000, timesteps=100, device='cpu'):
    initial_conditions = torch.randn(num_samples, 3, device=device) * 10  # Random initial conditions
    t = torch.linspace(0., 2., timesteps, device=device)
    data = odeint(lorenz_system, initial_conditions, t)
    return data.permute(1, 0, 2)  # Change to (timepoints, samples, state_size)

def visualize_data(data, num_trajectories=5):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    timesteps = data.shape[1]

    for i in range(min(num_trajectories, data.shape[0])):
        xs = data[i, :, 0].cpu().numpy()
        ys = data[i, :, 1].cpu().numpy()
        zs = data[i, :, 2].cpu().numpy()
        ax.plot(xs, ys, zs, lw=0.5)

    ax.set_title("Visualization of Lorenz System Trajectories")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    plt.show()

# Define the MLP model for the Lorenz system
class LorenzMLP(nn.Module):
    def __init__(self):
        super(LorenzMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 3)
        )
    
    def forward(self, t, y):
        return self.fc(y)

# Training the MLP
def train_model(data, epochs=1000, batch_size=64, device='cpu'):
    model = LorenzMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    t = torch.linspace(0., 2., data.shape[1], device=device)

    for epoch in range(epochs):
        permutation = torch.randperm(data.size(0))
        for i in range(0, data.size(0), batch_size):
            indices = permutation[i:i + batch_size].to(device)
            batch_x = data[indices].to(device)  # Shape: (batch_size, timepoints, state_size)
            batch_y = odeint(model, batch_x[:, 0, :], t)
            batch_y = batch_y.permute(1, 0, 2)  # Correct the shape to (batch_size, timepoints, state_size)

            loss = criterion(batch_y, batch_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    return model


# Plotting some results
def plot_results(model, data):
    t = torch.linspace(0., 2., data.shape[1])
    predicted = odeint(model, data[0, 0, :].to(data.device), t).detach()

    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    labels = ['X', 'Y', 'Z']
    for i in range(3):
        axs[i].plot(t.cpu().numpy(), data[0, :, i].cpu().numpy(), label='True')
        axs[i].plot(t.cpu().numpy(), predicted[:, i].cpu().numpy(), label='Predicted', linestyle='--')
        axs[i].set_ylabel(labels[i])
        axs[i].legend()
    plt.show()

# Main function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    data = generate_data(device=device)

    # Uncomment the next line to visualize data
    # visualize_data(data, num_trajectories=15)

    model = train_model(data, device=device)
    plot_results(model, data[:1])  # Plotting results for the first initial condition

if __name__ == "__main__":
    main()
