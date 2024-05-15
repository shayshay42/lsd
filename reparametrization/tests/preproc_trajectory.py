import scanpy as sc
import scvelo as scv
import numpy as np
from utils import self_avoiding_random_walk
from config import Config

adata = scv.datasets.pancreas()
scv.pp.filter_genes(adata, min_shared_counts=20)
scv.pp.normalize_per_cell(adata)
scv.pp.filter_genes_dispersion(adata, n_top_genes=2000)
sc.pp.log1p(adata)

sc.tl.diffmap(adata)
sc.pp.neighbors(adata, n_neighbors=30, use_rep='X_diffmap')
sc.tl.draw_graph(adata)

# Compute the pseudo-time
sc.tl.dpt(adata)

sc.pl.draw_graph(adata, color='dpt_pseudotime', title='Pseudo-time on Draw Graph')

