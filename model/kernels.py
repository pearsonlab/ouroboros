import numpy as np
from abc import ABC,abstractmethod
import torch
from torch import nn
from model.model_utils import smooth

class kernelModule(nn.Module):

    def __init__(self,nTerms,device,x_dim,z_dim,activation,trend_filtering):

        super().__init__()

        self.nTerms = nTerms
        self.device=device
        self.d = x_dim
        self.n = z_dim
        self.activation=activation
        self.trend_filtering = trend_filtering
        self.tau = 1
        pass

    @abstractmethod
    def forward(self,x):
        pass 

    def get_weights(self,z,smooth_len):

        return smooth(self.activation(self.weights(z)),smooth_len)
    
    def get_mus(self):

        return self.mus


class polyModule(kernelModule):

    def __init__(self,nTerms,device,x_dim,z_dim,lam=0.9,activation=nn.ReLU(),trend_filtering=True):

        super().__init__(nTerms,device,x_dim,z_dim,activation,trend_filtering)
        
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
        if not self.trend_filtering:
            weights = smooth(weights,smooth_len)
        power_mat = self.lam * (x[:,:,:,None].expand(-1,-1,-1,self.poly_dim-1) - self.mus)#[:,:,:,None].expand(-1,-1,-1,self.poly_dim+1)
        power_mat = torch.einsum('bldp,bldp->blp',power_mat,self.prods)[:,:,None,:]
        power_mat = power_mat.pow(self.powers)
        #weights = weights.view(B,L,d,self.poly_dim+1,self.poly_dim+1) * self.mask
        x = torch.einsum('blp,bldp->bld',weights,power_mat)
        #x = torch.einsum('bldj,bldj->bld',x,torch.flip(power_mat,[2])) 

    
        return x,weights

        
    def forward_given_weights(self,x,weights):

        B,L,d = x.shape
        if weights.shape != (B,L,self.d,self.poly_dim-1):
            weights = weights.view(B,L,self.d,self.poly_dim-1)
        #weights = weights.view(self.d,self.nTerms)

        power_mat = self.lam * (x[:,:,:,None].expand(-1,-1,-1,self.poly_dim-1) - self.mus)#[:,:,:,None].expand(-1,-1,-1,self.poly_dim+1)
        power_mat = torch.einsum('bldp,bldp->blp',power_mat,self.prods)[:,:,None,:]
        power_mat = power_mat.pow(self.powers)
        #weights = weights.view(B,L,d,self.poly_dim+1,self.poly_dim+1) * self.mask
        x = torch.einsum('blp,bldp->bld',weights,power_mat)
        #x = torch.einsum('bldj,bldj->bld',x,torch.flip(power_mat,[2])) 

        return x
    
    def forward_given_weights_numpy(self,x,weights):
        
        mus,prods = self.mus.detach().cpu().numpy(),self.prods.detach().cpu().numpy()
        powers = self.powers.detach().cpu().numpy()
        if len(x.shape)!=3:
            x = np.reshape(x,(x.shape[0],-1,2*self.d))
        B,L,d = x.shape
        if weights.shape != (B,L,self.d,self.poly_dim-1):
            weights = np.reshape(weights,(B,L,self.d,self.poly_dim-1))
        #weights = np.reshape(weights,(B,L,self.d,self.nTerms))
        power_mat = self.lam * (x[:,:,:,None].tile((1,1,1,self.poly_dim-1) - mus))
        power_mat = np.einsum('bldp,bldp->blp',power_mat,prods)[:,:,None,:]
        power_mat = np.power(power_mat,powers[None,None,:])
        
        x = np.einsum('blp,bldp->bld',weights,power_mat)
        
        return x
    
class fullPolyModule(kernelModule):

    def __init__(self,nTerms,device,x_dim,z_dim,lam=0.01,activation=nn.ReLU(),trend_filtering=True):

        super().__init__(nTerms,device,x_dim,z_dim,activation,trend_filtering)
        
        self.poly_dim = nTerms
        
        self.weights = nn.Linear(self.n,(self.poly_dim+1)**2).to(self.device)
        self.powers = torch.arange(0,self.poly_dim+1,device=self.device)
        self.lam = lam

    def forward(self,x,z,smooth_len):

        B,L,d = x.shape
        _,_,n = z.shape
        weights = self.activation(self.weights(z))
        if not self.trend_filtering:
            weights = smooth(weights,smooth_len)
        weights=weights.view(B,L,self.poly_dim+1,self.poly_dim+1)
        ### constant term
        weights[:,:,0,0] = weights[:,:,0,0] * 0 
        ### y, ydot terms
        weights[:,:,1,0] = weights[:,:,1,0] * 0 
        weights[:,:,0,1] = weights[:,:,0,1] * 0 
        power_mat = (x[:,:,:,None].expand(-1,-1,-1,self.poly_dim+1)).pow(self.powers)
        ## power mat is now: B x L x d x p
        ## we want to turn it into a B x L x 1 x p x p matrix
        
        z1 = power_mat[:,:,:1,:]
        z2 = power_mat[:,:,1:,:]
        power_mat = torch.einsum('bldp,bldk -> blpk',z1,z2)

        x = torch.einsum('blpd,blpd -> bl',weights,power_mat)
        #lam_mat = torch.full((B,L,self.poly_dim),self.lam,device=self.device).pow(self.powers)[:,:,:,None]
        #lam_mat = torch.arange(self.poly_dim+1,dtype=torch.float32,device=self.device)[None,None,:,None].expand(B,L,-1,self.poly_dim+1)
        #print(lam_mat.shape)
        #assert torch.all(lam_mat >= 0)
        #reg_weights = (lam_mat + lam_mat.transpose(-1,-2))*self.lam
        #assert np.all(reg_weights.shape == weights.shape) 
        #assert torch.all(reg_weights >= 0), print(torch.argwhere(reg_weights < 0))
        return x[:,:,None],weights


    def forward_given_weights(self,x,weights):

        B,L,d = x.shape
        weights=weights.view(B,L,self.poly_dim+1,self.poly_dim+1)
        ### constant term
        weights[:,:,0,0] = weights[:,:,0,0] * 0 
        ### y, ydot terms
        weights[:,:,1,0] = weights[:,:,1,0] * 0 
        weights[:,:,0,1] = weights[:,:,0,1] * 0 
        power_mat = (x[:,:,:,None].expand(-1,-1,-1,self.poly_dim+1)).pow(self.powers)

        z1 = power_mat[:,:,:1,:]
        z2 = power_mat[:,:,1:,:]
        power_mat = torch.einsum('bldp,bldk -> blpk',z1,z2)
        x = torch.einsum('blpd,blpd -> bl',weights,power_mat)
        #x = torch.einsum('bldj,bldj->bld',x,torch.flip(power_mat,[2])) 

        return x[:,:,None]
    
    def forward_given_weights_numpy(self,x,weights):
        
        powers = self.powers.detach().cpu().numpy()
        if len(x.shape)!=3:
            x = np.reshape(x,(x.shape[0],-1,2*self.d))
        B,L,d = x.shape
        #if weights.shape != (B,L,self.d,self.poly_dim-1):
        weights = np.reshape(weights,(B,L,self.poly_dim+1,self.poly_dim+1))

        # constant term
        weights[:,:,0,0] = weights[:,:,0,0] * 0 
        ### y, ydot terms
        weights[:,:,1,0] = weights[:,:,1,0] * 0 
        weights[:,:,0,1] = weights[:,:,0,1] * 0 

        power_mat = np.power(np.tile(x[:,:,:,None],(1,1,1,self.poly_dim+1)),powers)
        z1 = power_mat[:,:,:1,:]
        z2 = power_mat[:,:,1:,:]
        power_mat = np.einsum('blpd,bldk->blpk',z1,z2)
        x = np.einsum('blpd,blpd->bl',weights,power_mat)
        
        return x[:,:,None]


        
    
class fitPolyModule(polyModule):

    def __init__(self,nTerms,device,x_dim,z_dim,lam=0.9,activation=nn.ReLU(),trend_filtering=True):

        super().__init__(nTerms,device,x_dim,z_dim,lam,activation,trend_filtering)
        
        self.poly_dim = nTerms#*2*np.sqrt(2*self.d) - np.sqrt(2*self.d)
        self.mus = nn.Parameter(torch.rand((1,1,2*self.d,self.poly_dim-1),device=self.device),\
                                requires_grad=True)
        self.prods = nn.Parameter(torch.rand((1,1,2*self.d,self.poly_dim-1),device=self.device),\
                                requires_grad=True)

class simpleGaussModule(kernelModule):

    def __init__(self,nTerms,device,x_dim,z_dim,activation=nn.ReLU(),trend_filtering=True):

        super().__init__(nTerms,device,x_dim,z_dim,activation,trend_filtering)
        self.mus = nn.Parameter(torch.rand((1,1,2*self.d,nTerms),device=self.device),\
                                requires_grad=True)
        self.log_sigmas = nn.Parameter(torch.rand((1,1,2*self.d,nTerms),device=self.device),\
                                   requires_grad=True)
        self.weights = nn.Linear(self.n,self.d*self.nTerms).to(self.device)

    def forward(self,x,z,smooth_len):
        B,L,d = x.shape
        _,_,n = z.shape
        weights = self.activation(self.weights(z))
        if not self.trend_filtering:
            weights = smooth(weights,smooth_len)
        weights = weights.view(B,L,self.d,self.nTerms)
        
        gauss_mat = torch.linalg.norm((x[:,:,:,None].expand(-1,-1,-1,self.nTerms) - self.mus)/ (2*torch.exp(2*self.log_sigmas)),dim=2,keepdims=True)**2 
        kernels = torch.exp(-gauss_mat) #/(2*torch.pi * torch.exp(2*self.log_sigmas))**(d/2)

        x = torch.einsum('bldp,bldp->bld', weights,kernels)


        return x*self.tau,weights*self.tau
    
    def forward_given_weights(self,x,weights):

        B,L,d = x.shape
        if weights.shape != (B,L,self.d,self.nTerms):
            weights = weights.view(B,L,self.d,self.nTerms)

        gauss_mat = torch.linalg.norm((x[:,:,:,None].expand(-1,-1,-1,self.nTerms) - self.mus)/ (2*torch.exp(2*self.log_sigmas)),dim=2,keepdims=True)**2 
        kernels = torch.exp(-gauss_mat) #/(2*torch.pi * torch.exp(2*self.log_sigmas))**(d/2)

        x = torch.einsum('bldp,bldp->bld', weights,kernels)*self.tau

        return x
    
    def forward_given_weights_numpy(self,x,weights):
        # x must at least be Bx2D
        mus,log_sigmas = self.mus.detach().cpu().numpy(),self.log_sigmas.detach().cpu().numpy()
        if len(x.shape)!=3:
            x = np.reshape(x,(x.shape[0],-1,2*self.d))
        B,L,d = x.shape
        if weights.shape != (B,L,self.d,self.nTerms):
            weights = np.reshape(weights,(B,L,self.d,self.nTerms))
            
        gauss_mat = np.linalg.norm((np.tile(x[:,:,:,None],(1,1,1,self.nTerms)) - mus)/ (2*np.exp(2*log_sigmas)),axis=2,keepdims=True)**2
        kernels = np.exp(-gauss_mat)
        
        x = np.einsum('bldp,bldp->bld', weights,kernels)*self.tau
        
        return x
    
    def get_log_sigmas(self):

        return self.log_sigmas

class constantWeights(nn.Module):

    def __init__(self,dimension,nTerms,device='cuda'):

        super().__init__()

        self.d = dimension
        self.nTerms = nTerms
        self.device=device
        self.weights = nn.Parameter(torch.rand((1,1,self.d,nTerms),device=self.device),\
                                   requires_grad=True)
    
    def forward(self,x):

        B,L,_ = x.shape
        return self.weights.expand(B,L,-1,-1),
class constantGaussModule(simpleGaussModule):

    def __init__(self,nTerms,device,x_dim,z_dim,activation=nn.ReLU(),trend_filtering=True):

        super().__init__(nTerms,device,x_dim,z_dim,activation,trend_filtering)
        
        self.weights = constantWeights(self.d,self.nTerms).to(self.device)



