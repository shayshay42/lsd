import os
smoke_test = ('CI' in os.environ)  # for continuous integration tests

# various import statements
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

def prepare_data_loader(data, batch_size = 32, prior_time = False, labeled = False):
    norm_unspliced_tensor = torch.from_numpy(data.layers['normalized_unspliced'].toarray()).type(torch.float32)
    norm_spliced_tensor =  torch.from_numpy(data.layers['normalized_spliced'].toarray()).type(torch.float32)
    spliced_array = data.layers['spliced'].toarray().astype(np.int32)
    unspliced_array = data.layers['unspliced'].toarray().astype(np.int32)
    spliced_tensor = torch.from_numpy(spliced_array)
    unspliced_tensor = torch.from_numpy(unspliced_array)
    if prior_time:
        time = torch.from_numpy(data.obs['time'].values).type(torch.float32)
        if labeled:
            labels = torch.from_numpy(data.obs['labels'].values).type(torch.float32)
            dataset = TensorDataset(spliced_tensor, unspliced_tensor, norm_spliced_tensor, norm_unspliced_tensor, time, labels)
        else:
            dataset = TensorDataset(spliced_tensor, unspliced_tensor, norm_spliced_tensor, norm_unspliced_tensor, time)
    else:
        if labeled:
            labels = torch.from_numpy(data.obs['labels'].values).type(torch.float32)
            dataset = TensorDataset(spliced_tensor, unspliced_tensor, norm_spliced_tensor, norm_unspliced_tensor, labels)
        else:
            dataset = TensorDataset(spliced_tensor, unspliced_tensor, norm_spliced_tensor, norm_unspliced_tensor)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

#this function is used for calculating the MSE loss between z & z_hat
def calc_mse (z,t,vae):
    index = torch.argsort(t)
    original_index = torch.argsort(index)
    t_ode = t[index]
    IC = z[index][0]
    z_hat = odeint(vae.ode_func, IC, t_ode).squeeze()
    z_hat = z_hat[original_index]
    mse = nn.MSELoss()
    return mse(z, z_hat).item()