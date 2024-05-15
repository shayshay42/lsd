import torch
import torch.nn as nn
import torchdiffeq as tde
import torchode as to
import time

torch.manual_seed(0)

# Define the Lorenz system
def lorenz_system(t, y):
    sigma, beta, rho = 10.0, 8./3, 28.0
    x, y, z = y[..., 0], y[..., 1], y[..., 2]
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return torch.stack([dx, dy, dz], dim=-1)

# Generate data
def generate_data(num_samples=1000, timesteps=100, initial_scale=10.0):
    initial_conditions = torch.randn(num_samples, 3) * initial_scale
    t = torch.linspace(0., 2., timesteps)
    data = tde.odeint(lorenz_system, initial_conditions, t)
    return data.permute(1, 0, 2)

# Define the MLP model
class LorenzMLP(nn.Module):
    def __init__(self, hidden_size=50):
        super(LorenzMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 3)
        )
    
    def forward(self, t, y):
        return self.fc(y)


def train_model(data, use_torchode=True, epochs=10, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LorenzMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    t_base = torch.linspace(0., 2., data.shape[1], device=device)

    # Ensure the data tensor is on the correct device
    data = data.to(device)

    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        permutation = torch.randperm(data.size(0)).to(device)  # Ensure permutation is on the same device
        for i in range(0, data.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_x = data[indices]  # Now both data and indices are on the same device
            # Replicate `t_base` for each sample in the batch to match expected shape in torchode
            t_eval = t_base.unsqueeze(0).repeat(batch_x.shape[0], 1)

            if use_torchode:
                term = to.ODETerm(model)
                step_method = to.Dopri5(term=term)
                step_size_controller = to.IntegralController(atol=1e-9, rtol=1e-7, term=term)
                adjoint = to.AutoDiffAdjoint(step_method, step_size_controller)
                problem = to.InitialValueProblem(y0=batch_x[:, 0, :], t_eval=t_eval)
                batch_y = adjoint.solve(problem).ys.permute(1, 0, 2)  # Ensure proper permutation
            else:
                batch_y = tde.odeint(model, batch_x[:, 0, :], t_base).permute(1, 0, 2)

            # Explicitly ensure shapes match for the loss calculation
            if batch_y.shape != batch_x.shape:
                batch_y = batch_y.permute(1, 0, 2)  # Fix shape if not matching

            loss = criterion(batch_y, batch_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)

        average_epoch_loss = epoch_loss / data.size(0)
        if epoch % 1== 0:  # Print every 10 epochs
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_epoch_loss:.6f}")

    training_time = time.time() - start_time
    return model, training_time




# Main function
def main():
    data = generate_data(num_samples=500)
    model_tde, time_tde = train_model(data, use_torchode=False)
    print(f"Training time (torchdiffeq): {time_tde} seconds")

    model_to, time_to = train_model(data, use_torchode=True)
    print(f"Training time (torchode): {time_to} seconds")

    # You can add more detailed accuracy and prediction time comparison here.

if __name__ == "__main__":
    main()
