import torch
from torch import nn
from torch.nn.functional import softplus, softmax, sigmoid
from torch.distributions import constraints
import numpy as np

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions.util import broadcast_shape
from pyro.optim import MultiStepLR
from pyro.infer import SVI, config_enumerate, TraceEnum_ELBO
from pyro.contrib.examples.scanvi_data import get_data

from torchdiffeq import odeint_adjoint as odeint

def make_fc(dims):
    """
    Create a fully connected neural network with softplus activations.
    input : dims : list of dimensions of the neural network
    output : nn.Sequential object
    """
    layers = []
    for in_dim, out_dim in zip(dims, dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.Softplus())
    return nn.Sequential(*layers[:-1])  # Exclude final Softplus non-linearity


def split_in_half(t):
    """
    Split the tensor t in half along the final dimension
    separates mean and vairance dimensions
    """
    return t.reshape(t.shape[:-1] + (2, -1)).unbind(-2)


# Used in parameterizing p(s | z2)
#X for count
class XDecoder(nn.Module):
    # This __init__ statement is executed once upon construction of the neural network.
    # Here we specify that the neural network has input dimension z2_dim
    # and output dimension num_genes.
    def __init__(self, num_genes, z_dim, hidden_dims):
        super().__init__()
        # Create a list to store the layers
        dims = [z_dim] + hidden_dims + [2 * num_genes] # 2 * num_genes for ZINB for each gene theres 2 parameters
        self.fc = make_fc(dims)
    # This method defines the actual computation of the neural network. It takes
    # z2 as input and spits out two parameters that are then used in the model
    # to define the ZINB observation distribution. In particular it generates
    # `gate_logits`, which controls zero-inflation, and `mu` which encodes the
    # relative frequencies of different genes.
    def forward(self, z):
        gate, mu = split_in_half(self.fc(z))
        # Note that mu is normalized so that total count information is
        # encoded by the latent variable ℓ.
        mu = softmax(mu, dim=-1)
        # gate = sigmoid(gate)
        return gate, mu
    
# Used in parameterizing q(sl | s)
# L for library
class LEncoder(nn.Module):
    def __init__(self, num_genes, hidden_dims):
        super().__init__()
        dims = [num_genes] + hidden_dims + [2]
        self.fc = make_fc(dims)

    def forward(self, s):
        # Transform the counts x to log space for increased numerical stability.
        # Note that we only use this transformation here; in particular the observation
        # distribution in the model is a proper count distribution.
        l_loc, l_scale = split_in_half(self.fc(s))
        l_scale = softplus(l_scale)
        return l_loc, l_scale
    

# Used in parameterizing p(z2 | z1, s)
#X for count
class XEncoder(nn.Module):
    def __init__(self, z_dim,num_genes, hidden_dims):
        super().__init__()
        # 2*z_dim + 2  = 2* z_dim for latent reps and 2 for time distribution parameters
        dims = [2* num_genes] + hidden_dims + [2 * z_dim + 2]
        self.fc = make_fc(dims)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, u, s):
        u = u.type(torch.float32)
        s = s.type(torch.float32)
        x = torch.cat([s, u], dim=-1)
        # We reshape the input to be two-dimensional so that nn.BatchNorm1d behaves correctly
        x = x.reshape(-1, x.size(-1))
        hidden = self.fc(x)
        # If the input was three-dimensional we now restore the original shape
        hidden = hidden.reshape(x.shape[:-1] + hidden.shape[-1:])
        # t = hidden[...,-1]
        # hidden = hidden[...,:-1]
        loc, scale = split_in_half(hidden) # dims of z_dim + 1
        loc_z = loc[... , :-1] # dims of z_dim 
        loc_t = loc[... ,-1] # dims of 1
        scale_z = scale[... , :-1]
        scale_t = scale[... ,-1]
        
        # Here and elsewhere softplus ensures that scale is positive. Note that we generally
        # expect softplus to be more numerically stable than exp.
        scale_z = softplus(scale_z)
        scale_t = softplus(scale_t)
        loc_t = self.sigmoid(loc_t)
        return loc_z, scale_z, loc_t, scale_t
    
    
class ODEFunc(nn.Module):
    def __init__(self,z_dim = 10):
        super().__init__()
        self.latent_dim = z_dim
        # one idea is to use L-1 regularization to impose sparse connections
        self.f = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Softplus(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Softplus(),
            nn.Linear(self.latent_dim, self.latent_dim)
            )

    def forward(self, t, y):
        return self.f(y)


def prepare_vae(adata, mse_weight = 1, batch_size=128):
    num_genes = len(adata.var)

    # Calculate library size (total counts) for each cell
    spliced_library_size = np.log(adata.layers['spliced'].sum(axis=1))

    # Calculate mean and scale for spliced layer
    # priors for LEncoder class
    sl_mean = spliced_library_size.mean()
    sl_scale = spliced_library_size.std()

    # Calculate library size for unspliced layer
    unspliced_library_size = np.log(adata.layers['unspliced'].sum(axis=1))

    # Calculate mean and scale for unspliced layer
    ul_mean = unspliced_library_size.mean()
    ul_scale = unspliced_library_size.std()

    # Instantiate instance of model/guide and various neural networks
    vae = VAE(num_genes=num_genes,
              sl_loc=sl_mean, sl_scale=sl_scale, ul_loc=ul_mean, ul_scale=ul_scale, mse_weight = mse_weight,
              scale_factor=1.0 / (batch_size * num_genes))

    return vae


class VAE(nn.Module):
    def __init__(self, num_genes, sl_loc, sl_scale, ul_loc, ul_scale,
                 latent_dim=10, mse_weight = 1, scale_factor=1.0):
        self.num_genes = num_genes

        # This is the dimension of both z1 and z2
        self.latent_dim = latent_dim

        # The next two hyperparameters determine the prior over the log_count latent variable `l`
        self.ul_loc = ul_loc
        self.ul_scale = ul_scale
        self.sl_loc = sl_loc
        self.sl_scale = sl_scale

        # This hyperparameter controls the strength of the auxiliary classification loss
        # self.alpha = alpha
        self.scale_factor = scale_factor

        super().__init__()

        # Setup the various neural networks used in the model and guide
        self.x_encoder = XEncoder(z_dim=self.latent_dim, num_genes=self.num_genes,
                                    hidden_dims=[128 , 64])
        self.u_decoder = XDecoder(num_genes=num_genes, hidden_dims=[64, 128], z_dim=self.latent_dim)
        self.s_decoder = XDecoder(num_genes=num_genes, hidden_dims=[64, 128], z_dim=self.latent_dim)
        self.sl_encoder = LEncoder(num_genes=num_genes, hidden_dims=[64 , 32])
        self.ul_encoder = LEncoder(num_genes=num_genes, hidden_dims=[64 , 32])
        self.ode_func = ODEFunc(z_dim =  self.latent_dim)
        self.mse_weight = mse_weight
        self.epsilon = 0.003


    def model(self, u_raw, s_raw, u, s):
        # Register various nn.Modules (i.e. the decoder/encoder networks) with Pyro
        pyro.module("VAE", self)

        # This gene-level parameter modulates the variance of the observation distribution
        theta_u = pyro.param("inverse_dispersion_unspliced", 1000.0 * u_raw.new_ones(self.num_genes),
                           constraint=constraints.positive)
        theta_s = pyro.param("inverse_dispersion_spliced", 1000.0 * s_raw.new_ones(self.num_genes),
                           constraint=constraints.positive)

        # We scale all sample statements by scale_factor so that the ELBO loss function
        # is normalized wrt the number of datapoints and genes.
        # This helps with numerical stability during optimization.
        with pyro.plate("batch", len(u)), poutine.scale(scale=self.scale_factor):
            z = pyro.sample("z", dist.Normal(0, u.new_ones(self.latent_dim)).to_event(1))
            t = pyro.sample("t", dist.Normal(torch.zeros(1).cuda(),torch.ones(1).cuda()).to_event(0)) #event 0 meaning scalar?

            index = torch.argsort(t)
            original_index = torch.argsort(index)
            t_ode = t[index]
            IC = z[index][0]
            z_hat = odeint(self.ode_func, IC, t_ode).squeeze()
            z_hat = z_hat[original_index]
            mse = nn.MSELoss()
            mse_loss = mse(z_hat,z)
            
            
            gate_logits_u, mu_u = self.u_decoder(z)
            ul_scale = self.ul_scale * u.new_ones(1)
            ul = pyro.sample("ul", dist.LogNormal(self.ul_loc, ul_scale).to_event(1))
            rate_u = (ul * mu_u + self.epsilon).log() - (theta_u + self.epsilon).log()
            u_dist = dist.ZeroInflatedNegativeBinomial(gate_logits=gate_logits_u, total_count=theta_u,
                                                       logits=rate_u)
            pyro.sample("u", u_dist.to_event(1), obs=u_raw)


            sl_scale = self.sl_scale * s.new_ones(1)
            sl = pyro.sample("sl", dist.LogNormal(self.sl_loc, sl_scale).to_event(1))
            # Note that by construction mu is normalized (i.e. mu.sum(-1) == 1) and the
            # total scale of counts for each cell is determined by `l`
            gate_logits_s, mu_s = self.s_decoder(z)
            rate_s = (sl * mu_s + self.epsilon).log() - (theta_s + self.epsilon).log()
            s_dist = dist.ZeroInflatedNegativeBinomial(gate_logits=gate_logits_s, total_count=theta_s,
                                                       logits=rate_s)
            pyro.sample("s", s_dist.to_event(1), obs=s_raw)
            
            
            
            gate_logits_u_hat, mu_u_hat = self.u_decoder(z_hat)
            rate_u_hat = (ul * mu_u_hat + self.epsilon).log() - (theta_u + self.epsilon).log()
            u_hat_dist = dist.ZeroInflatedNegativeBinomial(gate_logits=gate_logits_u_hat, total_count=theta_u,
                                                       logits=rate_u_hat)
            pyro.sample("u_hat", u_hat_dist.to_event(1), obs=u_raw)

            

            gate_logits_s_hat, mu_s_hat = self.s_decoder(z_hat)
            rate_s_hat = (sl * mu_s_hat + self.epsilon).log() - (theta_s + self.epsilon).log()
            s_hat_dist = dist.ZeroInflatedNegativeBinomial(gate_logits=gate_logits_s_hat, total_count=theta_s,
                                                       logits=rate_s_hat)
            # Observe the datapoint x using the observation distribution x_dist
            pyro.sample("s_hat", s_hat_dist.to_event(1), obs=s_raw)
            
            pyro.factor('MSE_Loss', -torch.exp(self.mse_weight*mse_loss))
            

    # The guide specifies the variational distribution
    def guide(self, u_raw, s_raw, u , s):
        pyro.module("VAE", self)
        with pyro.plate("batch", len(u)), poutine.scale(scale=self.scale_factor):
            z_loc , z_scale, t_loc, t_scale = self.x_encoder(u,s)
            sl_loc, sl_scale = self.sl_encoder(s)
            pyro.sample("sl", dist.LogNormal(sl_loc, sl_scale).to_event(1))
            pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))
            pyro.sample("t", dist.Normal(t_loc, t_scale).to_event(0))
            ul_loc, ul_scale = self.ul_encoder(u)
            pyro.sample("ul", dist.LogNormal(ul_loc, ul_scale).to_event(1))


