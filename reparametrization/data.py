# %%
#import packages and utilities functions

import scanpy as sc
import scvelo as scv
import numpy as np
import torch
from utils import self_avoiding_random_walk
from config import Config

import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn as nn
from torch.nn.functional import softplus, softmax, sigmoid
from torch.distributions import constraints
from torch.optim import Adam
import torch.nn.functional as F


# %%
adata = scv.datasets.pancreas()

# %%
# adata= adata[adata.obs['clusters'] != cell_type_to_remove]
scv.pp.filter_genes(adata, min_shared_counts=20)
scv.pp.normalize_per_cell(adata)
scv.pp.filter_genes_dispersion(adata, n_top_genes=2000)
# scv.pp.neighbors(adata, n_neighbors=30)
spliced_library_size = adata.layers['spliced'].sum(axis=1)
unspliced_library_size = adata.layers['unspliced'].sum(axis=1)

temp_adata = adata.copy()
scv.pp.filter_genes(temp_adata, min_shared_counts=20)
scv.pp.normalize_per_cell(temp_adata)
scv.pp.filter_genes_dispersion(temp_adata, n_top_genes=2000)
sc.pp.log1p(temp_adata)
adata.layers['lognormalized_unspliced'] = temp_adata.layers['unspliced']
adata.layers['lognormalized_spliced'] = temp_adata.layers['spliced']

norm_unspliced_tensor = torch.from_numpy(adata.layers['lognormalized_unspliced'].toarray()).type(torch.float32)
norm_spliced_tensor =  torch.from_numpy(adata.layers['lognormalized_spliced'].toarray()).type(torch.float32)
spliced_tensor = torch.from_numpy(adata.layers['spliced'].toarray().astype(np.float32)).type(torch.float32)
unspliced_tensor = torch.from_numpy(adata.layers['unspliced'].toarray().astype(np.float32)).type(torch.float32)

dataset = TensorDataset(spliced_tensor, unspliced_tensor, norm_spliced_tensor, norm_unspliced_tensor)
data_loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
