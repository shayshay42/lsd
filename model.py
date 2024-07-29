# various import statements
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn as nn
from torch.nn.functional import softplus, softmax, sigmoid
from torch.distributions import constraints
from torch.optim import Adam
import torch.nn.functional as F

from torch.autograd import Variable
from torch.autograd.functional import jacobian

from torchdiffeq import odeint_adjoint as odeint

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions.util import broadcast_shape
from pyro.optim import MultiStepLR
from pyro.infer import SVI, config_enumerate, TraceEnum_ELBO
from pyro.contrib.examples.scanvi_data import get_data

import zuko
from pyro.contrib.zuko import ZukoToPyro

# Helper for making fully-connected neural networks
def make_fc(dims):
    layers = []
    for in_dim, out_dim in zip(dims, dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.Softplus())
    return nn.Sequential(*layers[:-1])  # Exclude final Softplus non-linearity
def make_f(dims):
    layers = []
    for in_dim, out_dim in zip(dims, dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.Softplus())
    return nn.Sequential(*layers[:-1])  # Exclude final Softplus non-linearity
# Splits a tensor in half along the final dimension
def split_in_half(t):
    return t.reshape(t.shape[:-1] + (2, -1)).unbind(-2)
# Used in parameterizing p(s | z2)
class XDecoder(nn.Module):
    # This __init__ statement is executed once upon construction of the neural network.
    # Here we specify that the neural network has input dimension z2_dim
    # and output dimension num_genes.
    def __init__(self, num_genes, z_dim, hidden_dims):
        super().__init__()
        # Create a list to store the layers
        dims = [z_dim] + hidden_dims + [2 * num_genes]
        self.fc = make_fc(dims)
    # This method defines the actual computation of the neural network. It takes
    # z2 as input and sMSELoss out two parameters that are then used in the model
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
class LEncoder(nn.Module):
    def __init__(self, num_genes, hidden_dims):
        super().__init__()
        dims = [num_genes] + hidden_dims + [2]
        self.fc = make_fc(dims)

    def forward(self, s):
        # Transform the counts x to log space for increased numerical stability.
        # Note that we only use this transformation here; in particular the observation
        # distribution in the model is a proper count distribution.
        s = torch.log(1+s)
        l_loc, l_scale = split_in_half(self.fc(s))
        l_scale = softplus(l_scale)
        return l_loc, l_scale
    
# Used in parameterizing p(z2 | z1, s)
class XEncoder(nn.Module):
    def __init__(self, z_dim,num_genes, hidden_dims, infer_time = False):
        super().__init__()
        if infer_time == False:
            dims = [2* num_genes] + hidden_dims + [2 * z_dim]
        else:
            dims = [2* num_genes] + hidden_dims + [2 * z_dim + 2]
        self.fc = make_fc(dims)
        self.sigmoid = nn.Sigmoid()
        self.infer_time = infer_time
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
        loc, scale = split_in_half(hidden)

        
        # Here and elsewhere softplus ensures that scale is positive. Note that we generally
        # expect softplus to be more numerically stable than exp.
        scale = softplus(scale)
        if self.infer_time == False:
            return loc, scale
        else:
            
            loc_z = loc[... , :-1]
            loc_t = loc[... ,-1]
            scale_z = scale[... , :-1]
            scale_t = scale[... ,-1]
            loc_t = self.sigmoid(loc_t)
            return loc_z, scale_z, loc_t, scale_t
# class PotentialNet(nn.Module):
#     def __init__(self,z_dim, hidden_dims):
#         super().__init__()
#         self.latent_dim = z_dim
#         dims = [z_dim] + hidden_dims + [1]
#         self.f = make_f(dims)

#     def forward(self, t, y):
#         return self.f(y)


# class ODEFunc(nn.Module):
#     def __init__(self, z_dim, hidden_dims):
#         super().__init__()
#         self.potential = PotentialNet(z_dim,hidden_dims)
    
#     def forward(self, t, y):
#         y.requires_grad_(True)
#         potential = self.potential(t = None, y = y)
#         grad = torch.autograd.grad(potential, y, grad_outputs=torch.ones_like(potential), create_graph=True)[0]
#         return -grad

class ODEFunc(nn.Module):
    def __init__(self,z_dim, hidden_dims):
        super().__init__()
        self.latent_dim = z_dim
        dims = [z_dim] + hidden_dims + [z_dim]
        self.f = make_f(dims)

    def forward(self, t, y):
        return self.f(y)
    
# Splits a tensor into three parts along the final dimension
def split_in_thirds(t):
    return t.reshape(t.shape[:-1] + (3, -1)).unbind(-2)


class PINN(nn.Module):
    def __init__(self,num_genes, hidden_dims, input_dim, output_dim):
        super().__init__()
        dims = [input_dim] + hidden_dims + [6*output_dim]
        self.fc = make_fc(dims)
    def forward(self, z):
        hidden = self.fc(z)
        # If the input was three-dimensional we now restore the original shape
        hidden = hidden.reshape(z.shape[:-1] + hidden.shape[-1:])
        loc , scale = split_in_half(hidden)
        alpha_loc, beta_loc, gamma_loc = split_in_thirds(loc)
        alpha_scale, beta_scale, gamma_scale = split_in_thirds(softplus(scale))
        
        
        return alpha_loc, beta_loc, gamma_loc,alpha_scale, beta_scale, gamma_scale

    
    
# Define the binary classifier model
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim*2) 
        self.fc2 = nn.Linear(input_dim*2, 1)         
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class VAE(nn.Module):
    
    def __init__(self, num_genes, sl_loc, sl_scale, ul_loc, ul_scale,
                 latent_dim=10, mse_weight = 1, scale_factor=1.0,pinn = False, infer_time = False, use_NF = False,
                 classification = False):
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
        self.infer_time = infer_time

        super().__init__()

        # Setup the various neural networks used in the model and guide
        self.x_encoder = XEncoder(z_dim=self.latent_dim, num_genes=self.num_genes,
                                    hidden_dims=[128 , 64], infer_time = infer_time)
        self.u_decoder = XDecoder(num_genes=num_genes, hidden_dims=[64, 128], z_dim=self.latent_dim)
        self.s_decoder = XDecoder(num_genes=num_genes, hidden_dims=[64, 128], z_dim=self.latent_dim)
        self.sl_encoder = LEncoder(num_genes=num_genes, hidden_dims=[64 , 32])
        self.ul_encoder = LEncoder(num_genes=num_genes, hidden_dims=[64 , 32])
        self.ode_func = ODEFunc(z_dim = self.latent_dim, hidden_dims=[10 , 10])
        self.mse_weight = mse_weight
        self.epsilon = 0.003
        if use_NF == True:
            self.prior = zuko.flows.MAF(
                features=self.latent_dim,
                transforms=3,
                hidden_features=(256, 256)
            )
        self.use_NF = use_NF
        self.use_pinn = pinn
        self.classification = classification
        if self.classification == True:
            self.classifier = BinaryClassifier(input_dim = self.latent_dim)
            
            

    def model(self, u_raw, s_raw, u, s, time, l =None):
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
            if self.infer_time == True:
                # law of total variance
                _ , __, t_loc, t_scale = self.x_encoder(u,s)
                var = torch.sqrt(torch.mean((t_loc- time)**2) + t_scale**2)
                t = pyro.sample("t", dist.Normal(time, var*u.new_ones(1)).to_event(0))
                index = torch.argsort(t)
                original_index = torch.argsort(index)
                t_ode = t[index]
            else :
                index = torch.argsort(time)
                original_index = torch.argsort(index)
                t_ode = time[index]
            
            if self.use_NF == True:
                z = pyro.sample("z", ZukoToPyro(self.prior()))
            else:
                z = pyro.sample("z", dist.Normal(0, u.new_ones(self.latent_dim)).to_event(1))
                
                
            IC = z[index][0]
            # z_hat = odeint(self.ode_func, IC, t_ode).squeeze().cuda()
            z_hat = odeint(self.ode_func, IC, t_ode).squeeze()
            z_hat = z_hat[original_index]
            if self.infer_time == True: 
                z_prime_loc , z_prime_scale, _, __ = self.x_encoder(u,s)
            else :
                z_prime_loc , z_prime_scale = self.x_encoder(u,s)
            z_prime = pyro.sample("z_prime", dist.Normal(z_prime_loc, z_prime_scale).to_event(1))
            
            mse = nn.MSELoss()
            ode_loss = mse(z_hat,z_prime)

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
            # coef = pyro.sample("coef", dist.LogNormal(torch.log(ode_loss),0.01).to_event(0))
            # pyro.factor('ODE_Loss', -torch.exp(ode_loss/coef))
            pyro.factor('ODE_Loss', -torch.exp(self.mse_weight*ode_loss))
            if self.classification == True:
                logits = self.classifier(z).squeeze()
                pyro.sample("obs", dist.Bernoulli(logits=logits), obs=l)


    # The guide specifies the variational distribution
    def guide(self, u_raw, s_raw, u , s, time, l =None):
        pyro.module("VAE", self)
        with pyro.plate("batch", len(u)), poutine.scale(scale=self.scale_factor):
            if self.infer_time == True: 
                z_loc , z_scale, t_loc, t_scale = self.x_encoder(u,s)
                pyro.sample("t", dist.Normal(t_loc, t_scale).to_event(0))
            else :
                z_loc , z_scale = self.x_encoder(u,s)
            sl_loc, sl_scale = self.sl_encoder(s)
            pyro.sample("sl", dist.LogNormal(sl_loc, sl_scale).to_event(1))
            pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))
            ul_loc, ul_scale = self.ul_encoder(u)
            pyro.sample("ul", dist.LogNormal(ul_loc, ul_scale).to_event(1))
