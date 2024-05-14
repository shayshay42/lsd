import numpy as np

def self_avoiding_random_walk(adjacency_matrix, start_node, walk_length):
    current_node = start_node
    walk = [current_node]
    
    num_nodes = adjacency_matrix.shape[0]  # Get the number of nodes from the shape attribute
    
    for _ in range(walk_length - 1):
        # Get the neighbors of the current node
        neighbors = np.where(adjacency_matrix[current_node] > 0)[0]
        # Calculate probabilities for transitioning to each neighbor
        neighbor_probs = adjacency_matrix[current_node, neighbors] / np.sum(adjacency_matrix[current_node, neighbors])
        # Choose the next node randomly based on probabilities
        next_node = np.random.choice(neighbors , p=neighbor_probs)
        
        # Append the next node to the walk
        walk.append(next_node)
        # Move to the next node
        current_node = next_node
    
    return walk


def calc_mse (z,t,vae):
    index = torch.argsort(t)
    original_index = torch.argsort(index)
    t_ode = t[index]
    IC = z[index][0]
    z_hat = odeint(vae.ode_func, IC, t_ode).squeeze()
    z_hat = z_hat[original_index]
    mse = nn.MSELoss()
    return mse(z, z_hat).item()