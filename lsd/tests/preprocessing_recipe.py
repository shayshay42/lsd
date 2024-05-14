import scanpy as sc
import scvelo as scv
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
adata = scv.datasets.pancreas()

# Create a copy for the zheng17 recipe processing
adata_zheng = adata.copy()

# Preprocess with scvelo's filter_and_normalize
scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
scv.pp.moments(adata, n_pcs=30, n_neighbors=30)

# Preprocess with scanpy's recipe_zheng17
sc.pp.recipe_zheng17(adata_zheng)
sc.tl.pca(adata_zheng, svd_solver='arpack')

# Visualization of Gene Expression Distributions
def plot_gene_expression(adata, title):
    sns.histplot(adata.X.flatten(), bins=50, kde=True)
    plt.title(title)
    plt.xlabel('Expression level')
    plt.ylabel('Frequency')
    plt.show()

plot_gene_expression(adata, 'Gene Expression Distribution After scvelo Preprocessing')
plot_gene_expression(adata_zheng, 'Gene Expression Distribution After Zheng17 Preprocessing')

# Statistical summary
print("Statistics with scvelo preprocessing:")
print(adata.to_df().describe().transpose()[['mean', 'std']])

print("\nStatistics with Zheng17 preprocessing:")
print(adata_zheng.to_df().describe().transpose()[['mean', 'std']])
