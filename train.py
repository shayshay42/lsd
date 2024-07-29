# We build a time resolved dataset out of without_MSE_Loss notebook
import os
smoke_test = ('CI' in os.environ)  # for continuous integration tests

# various import statements
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn as nn


from torchdiffeq import odeint_adjoint as odeint

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions.util import broadcast_shape
from pyro.optim import MultiStepLR
from pyro.infer import SVI, config_enumerate, TraceEnum_ELBO
from pyro.contrib.examples.scanvi_data import get_data
from torch.optim import Adam
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import scvelo as scv
import scanpy as sc
from utils import prepare_data_loader , calc_mse
from model import VAE

if torch.cuda.is_available():
    # Use GPU (CUDA)
    device = torch.device("cuda")
else:
    # Use CPU
    device = torch.device("cpu")
def prepare_vae(adata, latent_dim = 10, mse_weight = 1, batch_size=128 , infer_time = True, use_NF = False, classification = False):
    num_genes = len(adata.var)

    # Calculate library size (total counts) for each cell
    spliced_library_size = np.log(adata.layers['spliced'].sum(axis=1))

    # Calculate mean and scale for spliced layer
    sl_mean = spliced_library_size.mean()
    sl_scale = spliced_library_size.std()

    # Calculate library size for unspliced layer
    unspliced_library_size = np.log(adata.layers['unspliced'].sum(axis=1))

    # Calculate mean and scale for unspliced layer
    ul_mean = unspliced_library_size.mean()
    ul_scale = unspliced_library_size.std()

    # Instantiate instance of model/guide and various neural networks
    vae = VAE(num_genes=num_genes,
              sl_loc=sl_mean, sl_scale=sl_scale, ul_loc=ul_mean, ul_scale=ul_scale,latent_dim = latent_dim, mse_weight = mse_weight,
              scale_factor=1.0 / (batch_size * num_genes), infer_time = infer_time, use_NF = use_NF, classification= classification)

    return vae


from tqdm import tqdm
def train_vae(adata, latent_dim = 10, mse_weight = 1, num_epochs=80,batch_size = 128, smoke_test=False, infer_time = True,
              use_NF = False, labeled = False, plot_loss = True):
    # Clear Pyro param store so we don't conflict with previous
    # training runs in this session
    pyro.clear_param_store()

    # Fix random number seed
    pyro.util.set_rng_seed(42)

    # Enable optional validation warnings
    pyro.enable_validation(True)

    data_loader = prepare_data_loader(adata, batch_size, prior_time = True, labeled = labeled)
    num_genes = len(adata.var)
    vae = prepare_vae(adata,latent_dim, mse_weight, batch_size, infer_time,
                      use_NF = use_NF, classification = labeled)
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
        if infer_time == True :
            time_var_loss = []

        # Take a gradient step for each mini-batch in the dataset
        for batch in tqdm(data_loader, desc=f'Epoch {epoch}'):
            if labeled == False:
                
                s_raw, u_raw, s, u, t = batch
                s_raw, u_raw, s, u, t = s_raw.to(device), u_raw.to(device), s.to(device), u.to(device), t.to(device)
            else:
                s_raw, u_raw, s, u, t, l = batch
                s_raw, u_raw, s, u, t, l = s_raw.to(device), u_raw.to(device), s.to(device), u.to(device), t.to(device), l.to(device)
            
            enc = vae.x_encoder(u, s)
            z = pyro.sample("z", dist.Normal(enc[0], enc[1]).to_event(1))
            if infer_time == True: 
                time = enc[2]
                l2loss = calc_mse(z,time,vae)
                mse = nn.MSELoss()
                timeloss = mse(time, t) 
                time_var_loss.append(timeloss.item())
            else: 
                l2loss = calc_mse(z,t,vae)
            if labeled == False:
                loss = svi.step(u_raw, s_raw, u, s, t)
            else:
                loss = svi.step(u_raw, s_raw, u, s, t, l)
            epoch_losses.append(loss)
            l2_losses.append(l2loss)

        # Tell the scheduler we've done one epoch.
        scheduler.step()

        if plot_loss ==True:
            plt.figure(figsize=(10, 5))
            plt.plot(l2_losses, label='L2 Loss')
            plt.xlabel('Iter')
            plt.ylabel('ODE L2 Loss')
            plt.legend()
            plt.title('L2 Loss Over Epochs')
            plt.show()
            if infer_time == True:

                plt.figure(figsize=(10, 5))
                plt.plot(time_var_loss, label='Time Variance Loss')
                plt.xlabel('Iter')
                plt.ylabel('t - real_t')
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