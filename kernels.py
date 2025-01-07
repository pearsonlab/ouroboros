import numpy as np
from abc import ABC,abstractmethod
import torch
from torch import nn
from model_utils import smooth

class kernelModule(nn.Module):

    def __init__(self,nTerms,device,x_dim,z_dim,activation):

        super().__init__()

        self.nTerms = nTerms
        self.device=device
        self.d = x_dim
        self.n = z_dim
        self.activation=activation
        pass

    @abstractmethod
    def forward(self,x):
        pass 

    def get_weights(self,z):

        return self.activation(self.weights(z))
    
    def get_mus(self):

        return self.mus


class polyModule(kernelModule):

    def __init__(self,nTerms,device,x_dim,z_dim,lam=0.9,activation=nn.ReLU()):

        super().__init__(nTerms,device,x_dim,z_dim,activation)
        
        self.poly_dim = nTerms
        self.mus = torch.zeros((1,1,2*self.d,self.poly_dim-1),device=self.device)
                                #nn.Parameter(torch.rand((1,1,2*self.d,self.poly_dim-1),device=self.device)*2*np.sqrt(2*self.d) - np.sqrt(2*self.d),\
                   #             requires_grad=True)
        self.prods = torch.ones((1,1,2*self.d,self.poly_dim-1),device=self.device)
        #nn.Parameter(torch.rand((1,1,2*self.d,self.poly_dim-1),device=self.device)*2*np.sqrt(2*self.d) - np.sqrt(2*self.d),\
        #                        requires_grad=True)
        self.weights = nn.Linear(self.n,self.poly_dim-1).to(self.device)
        self.powers = torch.arange(2,self.poly_dim+1,device=self.device)
        self.lam = lam

    def forward(self,x,z,smooth_len):

        B,L,d = x.shape
        _,_,n = z.shape
        weights = self.activation(self.weights(z))
        weights = smooth(weights,smooth_len)
        power_mat = self.lam * (x[:,:,:,None].expand(-1,-1,-1,self.poly_dim-1) - self.mus)#[:,:,:,None].expand(-1,-1,-1,self.poly_dim+1)
        power_mat = torch.einsum('bldp,bldp->blp',power_mat,self.prods)[:,:,None,:]
        power_mat = power_mat.pow(self.powers)
        #weights = weights.view(B,L,d,self.poly_dim+1,self.poly_dim+1) * self.mask
        x = torch.einsum('blp,bldp->bld',weights,power_mat)
        #x = torch.einsum('bldj,bldj->bld',x,torch.flip(power_mat,[2])) 

        return x
    
class fitPolyModule(polyModule):

    def __init__(self,nTerms,device,x_dim,z_dim,lam=0.9,activation=nn.ReLU()):

        super().__init__(nTerms,device,x_dim,z_dim,lam,activation)
        
        self.poly_dim = nTerms
        self.mus = nn.Parameter(torch.rand((1,1,2*self.d,self.poly_dim-1),device=self.device)*2*np.sqrt(2*self.d) - np.sqrt(2*self.d),\
                                requires_grad=True)
        self.prods = nn.Parameter(torch.rand((1,1,2*self.d,self.poly_dim-1),device=self.device)*2*np.sqrt(2*self.d) - np.sqrt(2*self.d),\
                                requires_grad=True)

class simpleGaussModule(kernelModule):

    def __init__(self,nTerms,device,x_dim,z_dim,activation=nn.ReLU()):

        super().__init__(nTerms,device,x_dim,z_dim,activation)
        self.mus = nn.Parameter(torch.rand((1,1,self.d,nTerms),device=self.device)*2*np.sqrt(self.d*nTerms) - np.sqrt(self.d *nTerms),\
                                requires_grad=True)
        self.log_sigmas = nn.Parameter(torch.rand((1,1,1,nTerms),device=self.device)*2*np.sqrt(self.nTerms) - np.sqrt(self.nTerms),\
                                   requires_grad=True)
        self.weights = nn.Linear(self.n,self.d*self.nTerms).to(self.device)

    def forward(self,x,z,smooth_len):
        B,L,d = x.shape
        _,_,n = z.shape
        weights = self.activation(self.weights(z))
        weights = smooth(weights,smooth_len)
        weights = weights.view(B,L,self.d,self.nTerms)
        gauss_mat = torch.linalg.norm(x[:,:,:,None].expand(-1,-1,-1,self.nTerms) - self.mus,dim=2,keepdims=True)**2 / (2*torch.exp(2*self.log_sigmas))
        kernels = torch.exp(-gauss_mat) #/(2*torch.pi * torch.exp(2*self.log_sigmas))**(d/2)

        x = torch.einsum('bldp,bldp->bld', weights,kernels)

        return x
    
    def get_log_sigmas(self):

        return self.log_sigmas
