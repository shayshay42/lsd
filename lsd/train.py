from config import Config
from data import load_data
from model import create_model

import pyro
import torch


from tqdm import tqdm
def train_vae(adata, mse_weight = 1, num_epochs=80,batch_size = 128, smoke_test=False):
    # Clear Pyro param store so we don't conflict with previous
    # training runs in this session
    pyro.clear_param_store()

    # Fix random number seed
    pyro.util.set_rng_seed(42)

    # Enable optional validation warnings
    pyro.enable_validation(True)

    dataset = TensorDataset(spliced_tensor, unspliced_tensor, norm_spliced_tensor, norm_unspliced_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num_genes = len(adata.var)
    vae = prepare_vae(adata, mse_weight, batch_size)
    if not smoke_test:
        if torch.cuda.is_available():
            # Use GPU (CUDA)
            device = torch.device("cuda")
        else:
            # Use CPU
            device = torch.device("cpu")
        vae = vae.to(device)
     # Setup an optimizer (Adam) and learning rate scheduler.
    # We start with a moderately high learning rate (0.006) and
    # reduce by a factor of 5 after 20 epochs.
    scheduler = MultiStepLR({'optimizer': Adam,
                             'optim_args': {'lr': 0.006},
                             'gamma': 0.2, 'milestones': [5]},
                              {"clip_norm": 10.0})

    # Setup a variational objective for gradient-based learning.
    # Note we use TraceEnum_ELBO in order to leverage Pyro's machinery
    # for automatic enumeration of the discrete latent variable y.
    elbo = TraceEnum_ELBO(strict_enumeration_warning=False)
    svi = SVI(vae.model, vae.guide, scheduler, elbo)

    # Training loop.
    # We train for num_epochs epochs.
    # For optimal results, tweak the optimization parameters.
    # For our purposes, 80 epochs of training is sufficient.
    # Training should take about 8 minutes on a GPU-equipped Colab instance.

    losses = []

    for epoch in range(num_epochs):
        epoch_losses = []
        l2_losses = []

        # Take a gradient step for each mini-batch in the dataset
        for s_raw, u_raw, s ,u in tqdm(data_loader, desc=f'Epoch {epoch}'):
            s_raw, u_raw, s, u = s_raw.to(device), u_raw.to(device), s.to(device), u.to(device)
            loss = svi.step(u_raw, s_raw, u, s)
            epoch_losses.append(loss)
            enc = vae.x_encoder(u, s)
            z = pyro.sample("z", dist.Normal(enc[0], enc[1]).to_event(1))
            t = pyro.sample("t", dist.Normal(enc[2], enc[3]).to_event(0))
            l2loss = calc_mse(z,t,vae)
            l2_losses.append(l2loss)

        # Tell the scheduler we've done one epoch.
        scheduler.step()

        
        plt.figure(figsize=(10, 5))
        plt.plot(l2_losses, label='L2 Loss')
        plt.xlabel('Iter')
        plt.ylabel('ODE L2 Loss')
        plt.legend()
        plt.title('L2 Loss Over Epochs')
        plt.show()
        epoch_loss_mean = np.mean(epoch_losses)
        losses.append(epoch_loss_mean)

        if epoch%1 ==0:
            print(f"[Epoch {epoch}]  Loss: {epoch_loss_mean:.5f}")


    print("Finished training!")

    # Plot the loss function
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.close() 

    return vae


if torch.cuda.is_available():
    # Use GPU (CUDA)
    device = torch.device("cuda")
else:
    # Use CPU
    device = torch.device("cpu")


trained_vae = train_vae(adata, mse_weight= 1, 
                            num_epochs=5, batch_size=8)