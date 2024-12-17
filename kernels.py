import numpy as np
from abc import ABC,abstractmethod
import torch
from torch import nn

class kernelModule(nn.Module):

    def __init__(self,nTerms,device,x_dim,z_dim):

        super().__init__()

        self.nTerms = nTerms
        self.device=device
        self.d = x_dim
        self.n = z_dim
        pass

    @abstractmethod
    def forward(self,x):
        pass 

    def get_weights(self,z):

        return self.weights(z)
    
    def get_mus(self):

        return self.mus


class polyModule(kernelModule):

    def __init__(self,nTerms,device,x_dim,z_dim):

        super().__init__(nTerms,device,x_dim,z_dim)
        self.mus = nn.Parameter(torch.rand((1,1,2*self.d,),device=self.device)*2*np.sqrt(2*self.d) - np.sqrt(2*self.d),\
                                requires_grad=True)
        self.poly_dim = nTerms
        self.weights = nn.Linear(self.n,2*self.d*(nTerms+1)**2).to(self.device)
        self.powers = torch.arange(0,nTerms+1,device=self.device)
        self.poly_dim = nTerms
        self.mask = torch.ones((2*self.d,nTerms+1,nTerms+1))
        self.mask[:,0,0] = 0
        self.mask[:,0,1] = 0
        self.mask[:,1,0] = 0
        self.mask = self.mask.to(self.device)


    def forward(self,x,z):

        B,L,d = x.shape
        _,_,n = z.shape
        weights = self.weights(z)
        power_mat = (x - self.mus)[:,:,:,None].expand(-1,-1,-1,self.poly_dim+1)
        power_mat = power_mat.pow(self.powers)
        weights = weights.view(B,L,d,self.poly_dim+1,self.poly_dim+1) * self.mask
        x = torch.einsum('bldkj,bldk->bldj',weights,power_mat)
        x = torch.einsum('bldj,bldj->bld',x,torch.flip(power_mat,[2])) 

        return x[:,:,self.d:]
    
class simpleGaussModule(kernelModule):

    def __init__(self,nTerms,device,x_dim,z_dim):

        super().__init__(nTerms,device,x_dim,z_dim)
        self.mus = nn.Parameter(torch.rand((1,1,self.d,nTerms),device=self.device)*2*np.sqrt(self.d*nTerms) - np.sqrt(self.d *nTerms),\
                                requires_grad=True)
        self.log_sigmas = nn.Parameter(torch.rand((1,1,1,nTerms),device=self.device)*2*np.sqrt(self.nTerms) - np.sqrt(self.nTerms),\
                                   requires_grad=True)
        self.weights = nn.Linear(self.n,self.d*self.nTerms).to(self.device)

    def forward(self,x,z):
        B,L,d = x.shape
        _,_,n = z.shape
        weights = self.weights(z)
        weights = weights.view(B,L,self.d,self.nTerms)
        gauss_mat = torch.linalg.norm(x[:,:,:,None].expand(-1,-1,-1,self.nTerms) - self.mus,dim=2,keepdims=True)**2 / (2*torch.exp(self.log_sigmas))
        kernels = torch.exp(gauss_mat)/(2*torch.pi * torch.exp(2*self.log_sigmas))**(d/2)

        x = torch.einsum('bldp,bldp->bld', weights,kernels)

        return x
    
    def get_log_sigmas(self):

        return self.log_sigmas
