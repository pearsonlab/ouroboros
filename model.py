from mambapy.mamba import Mamba, MambaConfig

import torch
import torch.nn as nn
from torchaudio.functional import lowpass_biquad
from torchdiffeq import odeint
import numpy as np
from scipy.integrate import solve_ivp
from utils import deriv_approx_dy,deriv_approx_d2y

torch.set_default_dtype(torch.float64)
def corr_fnc(ts1,tsSet):

    sd1,sd2 = ts1.std(dim=1,keepdim=True), tsSet.std(dim=1,keepdim=True)
    batch_denom = sd1.transpose(2,1) @ sd2

    batch_cov = ts1.transpose(2,1) @ tsSet/(ts1.shape[1]-1)

    batch_corr = batch_denom/batch_cov
    #mean_corrs = batch_corr.mean(dim=0)

    #inds = torch.triu_indices(mean_corrs.shape[0],mean_corrs.shape[1])
    
    return batch_corr.mean(dim=0).sum()


vmap_corr = torch.vmap(corr_fnc,in_dims=(2,None))


class ouroboros(nn.Module):


    def __init__(self,d_data,
                 d_control,
                 d_out=1,
                 n_layers_data=2,
                 n_layers_control=2,
                 d_state_data=16,
                 d_state_control=16,
                 d_conv_data=4,
                 d_conv_control=4,
                 expand_factor_data=1,
                 expand_factor_control=1,
                 device='cuda'):
        
        super(ouroboros,self).__init__()

        dataConfig = MambaConfig(d_model=d_out,\
                                    n_layers=n_layers_data,d_state=d_state_data,\
                                    d_conv=d_conv_data,expand_factor=expand_factor_data)
        controlConfig = MambaConfig(d_model=d_control, n_layers=n_layers_control,\
                                    d_state=d_state_control,d_conv=d_conv_control,\
                                    expand_factor=expand_factor_control)
        
        self.dataMamba = Mamba(dataConfig).to(device)
        self.controlMamba = Mamba(controlConfig).to(device)

        self.outProj = nn.Linear(d_control,d_out).to(device)
        self.control_proj = nn.Linear(d_data*4,d_control).to(device)
        self.device = device


    def forward(self,x,y):

        dy = y - x

        # CHANGE: PROJECT TO CONTROL DIM BEFORE GOING INTO CONTROL MAMBA
        # CHANGE: double flippy
        x_in = torch.cat([dy,y],dim=-1)
        x_in = torch.cat([x_in, torch.flip(x_in,[1])],dim=-1)
        state_pred = self.controlMamba(self.control_proj(x_in)) 
        state_pred = torch.flip(torch.nn.SiLU()(state_pred),[1]) # bsz x L x dim


        # CHANGE: PROJECT TO OUT DIM BEFORE GOING INTO DATA MAMBA
        yhat = self.outProj(state_pred)
        yhat = self.dataMamba(yhat)
        #yhat = self.outProj(yhat)

        ## penalize correlation between states?

        return yhat, state_pred
    
class timescale_ouroboros(ouroboros):


    def __init__(self,d_data,
                 d_control,
                 fs,
                 control_time_constant=0.05, # time constant of control signal in s 
                 n_layers_data=2,
                 n_layers_control=2,
                 d_state_data=16,
                 d_state_control=16,
                 d_conv_data=4,
                 d_conv_control=4,
                 expand_factor_data=1,
                 expand_factor_control=1,
                 device='cuda'):
        
        super(timescale_ouroboros,self).__init__(d_data,
                                                d_control=d_control,
                                                n_layers_data=n_layers_data,
                                                n_layers_control=n_layers_control,
                                                d_state_data=d_state_data,
                                                d_state_control=d_state_control,
                                                d_conv_data=d_conv_data,
                                                d_conv_control=d_conv_control,
                                                expand_factor_data=expand_factor_data,
                                                expand_factor_control=expand_factor_control,
                                                device=device)

        self.fs = fs
        self.control_time_constant_s = control_time_constant
        self.control_time_constant_fs = fs * control_time_constant
        

        self.control_init_conditions = nn.Linear(d_data*2,d_control).to(device)


    def forward(self,x,y,mask_ind = -1):

        dy = y - x

    
        state_ic = torch.nn.SiLU()(self.control_init_conditions(torch.cat([dy[:,:1,:],y[:,:1,:]],dim=-1)))
        assert state_ic.shape[1] == 1,print(state_ic.shape)
        state_pred = self.controlMamba(torch.cat([dy,y],dim=-1))
        state_pred = self.control_proj(state_pred)
        state_pred = state_ic + torch.cumsum(state_pred,dim=1)/self.control_time_constant_fs

        yhat = self.dataMamba(torch.cat([state_pred,x],dim=-1))

        return yhat, state_pred
    
class filter_ouroboros(ouroboros):

    def __init__(self,d_data,
                 d_control,
                 d_out=1,
                 n_layers_data=2,
                 n_layers_control=2,
                 d_state_data=16,
                 d_state_control=16,
                 d_conv_data=4,
                 d_conv_control=4,
                 expand_factor_data=1,
                 expand_factor_control=1,
                 device='cuda',
                 freq_limit=100,
                 fs=1000):
        
        super(filter_ouroboros,self).__init__(d_data,\
                 d_control,\
                 d_out,\
                 n_layers_data,\
                 n_layers_control,\
                 d_state_data,\
                 d_state_control,\
                 d_conv_data,\
                 d_conv_control,\
                 expand_factor_data,\
                 expand_factor_control,\
                 device) 
        
        self.freq_limit = freq_limit
        self.fs=fs
        self.filter = torch.vmap(lambda x: lowpass_biquad(x,self.fs,self.freq_limit),in_dims=2,out_dims=2)
        print("warning: batched backprop through filter not implemented in torch, this will be slow")

    def forward(self,x,y):
        

        dy = y - x

        state_pred = self.controlMamba(torch.flip(torch.cat([dy,y],dim=-1),[1]))
        state_pred = torch.flip(torch.nn.SiLU()(self.control_proj(state_pred)),[1]) # bsz x L x dim
        state_pred = self.filter(state_pred)

        yhat = self.dataMamba(state_pred)
        yhat = self.outProj(yhat)


        return yhat, state_pred



class fancy_ouroboros(ouroboros):

    def __init__(self,d_data,
                 d_control,
                 d_out=1,
                 n_layers_data=2,
                 n_layers_control=2,
                 d_state_data=16,
                 d_state_control=16,
                 d_conv_data=4,
                 d_conv_control=4,
                 expand_factor_data=1,
                 expand_factor_control=1,
                 device='cuda'):
        
        super(fancy_ouroboros,self).__init__(d_data,\
                 d_control,\
                 d_out,\
                 n_layers_data,\
                 n_layers_control,\
                 d_state_data,\
                 d_state_control,\
                 d_conv_data,\
                 d_conv_control,\
                 expand_factor_data,\
                 expand_factor_control,\
                 device)
        

        dataConfig = MambaConfig(d_model=d_control,\
                                    n_layers=n_layers_data,d_state=d_state_data,\
                                    d_conv=d_conv_data,expand_factor=expand_factor_data)
        
        self.dataMamba = Mamba(dataConfig).to(device)
        self.omega = nn.Linear(d_control,d_out*2).to(device)
        self.inp = nn.Linear(d_control,d_out).to(device)
        self.d_out = d_out

    def forward(self,x,y):

        xdot= y - x

        state_pred = self.controlMamba(self.control_proj(torch.flip(torch.cat([xdot,y],dim=-1),[1])))
        state_pred = torch.flip(torch.nn.SiLU()(state_pred),[1])

        out = self.dataMamba(state_pred)
        omega = self.omega(out)
        inp = self.inp(out)

        omega_terms = -omega*x #+ inp

        xdotdothat = torch.nn.ReLU()(omega_terms[:,:,:self.d_out]) + inp
        xdothat = omega_terms[:,:,self.d_out:]

        return torch.cat([xdothat,xdotdothat],dim=-1)[:,1:,],state_pred
    

class structured_ouroboros(ouroboros):



    def __init__(self,d_data,
                 d_control,
                 d_out=1,
                 n_layers_data=2,
                 n_layers_control=2,
                 d_state_data=16,
                 d_state_control=16,
                 d_conv_data=4,
                 d_conv_control=4,
                 expand_factor_data=1,
                 expand_factor_control=1,
                 device='cuda',
                 poly_dim=40):
        
        super(structured_ouroboros,self).__init__(d_data,\
                 d_control,\
                 d_out,\
                 n_layers_data,\
                 n_layers_control,\
                 d_state_data,\
                 d_state_control,\
                 d_conv_data,\
                 d_conv_control,\
                 expand_factor_data,\
                 expand_factor_control,\
                 device)
        
        self.b_net = nn.Linear(in_features=d_control,out_features=2*d_data * poly_dim**2,device=device)
        
        self.poly_dim = poly_dim
        self.d_data = d_data 
        self.powers = torch.arange(0,poly_dim,device=device)

    def forward(self,x,y):

        B,L,d = x.shape
        xdot= y - x
        z = torch.cat([x,xdot],dim=-1)
        power_mat = z[:,:,:,None].expand(-1,-1,-1,self.poly_dim) # B x L x 2d -> B x L x 2d x P
        power_mat = power_mat.pow(self.powers)
        

        state_pred = self.controlMamba(self.control_proj(torch.flip(torch.cat([xdot,y],dim=-1),[1])))
        state_pred = torch.flip(state_pred,[1]) 

        b = self.b_net(state_pred).view(B,L,2*d,self.poly_dim,self.poly_dim) # B x L x 2d x P x P
        pow1 = torch.einsum('bldkj,bldk->bldj',b,power_mat)
        zdot = torch.einsum('bldj,bldj->bld',pow1,torch.flip(power_mat,[2])) 
        
        return zdot[:,1:,:],state_pred

    def get_funcs(self,x,y):

        B,L,d = x.shape
        xdot= y - x
        z = torch.cat([x,xdot],dim=-1)
        power_mat = z[:,:,:,None].expand(-1,-1,-1,self.poly_dim) # B x L x 2d -> B x L x 2d x P
        power_mat = power_mat.pow(self.powers)
        

        state_pred = self.controlMamba(self.control_proj(torch.flip(torch.cat([xdot,y],dim=-1),[1])))
        state_pred = torch.flip(state_pred,[1]) #torch.flip(torch.nn.SiLU()(state_pred),[1])

        b = self.b_net(state_pred).view(B,L,2*d,self.poly_dim,self.poly_dim) # B x L x 2d x P x P
        
        A = torch.stack([b[:,:,:,1,0].detach().clone(),torch.flip(b,[2])[:,:,:,0,1].detach().clone()],dim=-1) #b[:,:,:,[0,1],[0,1]]
        print(A.shape)

        c = b[:,:,:,0,0].detach().clone()

        b[:,:,:,0,0] = 0
        #b[:,:,:,1,0] = 0
        #b[:,:,:,0,1] = 0
        
        pow1 = torch.einsum('bldkj,bldk->bldj',b,power_mat) # B x L x 2d x P 
        #pow2 = torch.einsum('bldkj,bldj->bldk',b,torch.flip(power_mat,[2]))
        pow2 = torch.einsum('bldj,bldj->bld',pow1,torch.flip(power_mat,[2])) # B x L x 2d x P

        b_out = pow2#torch.einsum('bldj,bldj->bld',pow1,pow2)

        
        return A,b,b_out,c

    def integrate(self,data):

        x,y = data[:,:-1,:],data[:,1:,:]
        B,L,d = x.shape
        dx = y - x 
        z = torch.cat([x,dx],dim=-1)
        z0 = z[:,0,:]


        state_pred = self.controlMamba(self.control_proj(torch.flip(torch.cat([dx,y],dim=-1),[1])))
        state_pred = torch.flip(state_pred,[1])
        # y contains: x,dx,caches (for mamba)
        b = self.b_net(state_pred).view(B,L,2*d,self.poly_dim,self.poly_dim)

        x,caches = y
        
        # for initialization
        #caches = [(None, torch.zeros((1,model.config.d_model * model.config.expand_factor,model.config.d_conv),device='cuda')) for _ in model.layers]
        #B,L,d = x.shape
        
        z = torch.cat([x,dx],dim=-1)
        state_pred,caches = self.controlMamba.step(self.control_proj(torch.flip(torch.cat([x[:,1],x[:,0]+x[:,1]],dim=-1),[1])))

        state_pred = torch.flip(state_pred,[1]) #torch.flip(torch.nn.SiLU()(state_pred),[1])

        #out = self.outProj(state_pred)
        #A = self.A_net(state_pred).view(2*d,2*d) # 1 x 2d x 2d
        #A = torch.einsum('dk,k->d',A,z)
        b = self.b_net(state_pred).view(2*d,self.poly_dim,self.poly_dim) # 1 x 2d x P x P
        pow1 = torch.einsum('dkj,dk->dj',b,self.power_mat)
        pow2 = torch.einsum('dkj,dj->dk',pow1,torch.flip(self.power_mat,[2]))
        b = torch.einsum('dj,dj->d',pow1,pow2)

        #c= self.c_net(state_pred)

        zdot = b#A + b + c
        return (zdot,caches)
        

class NonNegClipper(object):

    def __init__(self,min=0):
        self.min=0

    def __call__(self,module):

        if hasattr(module,'weight'):
            w = module.weight.data
            w.clamp_(min=self.min)
        if hasattr(module,'bias'):
            b = module.bias.data
            b.clamp_(min=self.min)

#plt.rcParams.update({'text.usetex':True})
class constrained_ouroboros(ouroboros):



    def __init__(self,d_data,
                 d_control,
                 d_out=1,
                 n_layers_data=2,
                 n_layers_control=2,
                 d_state_data=16,
                 d_state_control=16,
                 d_conv_data=4,
                 d_conv_control=4,
                 expand_factor_data=1,
                 expand_factor_control=1,
                 device='cuda',
                 poly_dim=40,
                good_init=True,
                tau = 1000,
                omega=4000,
                noInput = False,
                omega_scale = 10000,
                smooth_penalty=lambda x: torch.var(x,dim=1)):
        
        super(constrained_ouroboros,self).__init__(d_data,\
                 d_control,\
                 d_out,\
                 n_layers_data,\
                 n_layers_control,\
                 d_state_data,\
                 d_state_control,\
                 d_conv_data,\
                 d_conv_control,\
                 expand_factor_data,\
                 expand_factor_control,\
                 device)

        self.clipper = NonNegClipper(min=0)
        self.smooth_penalty = smooth_penalty
        #self.b_net = nn.Linear(in_features=d_control,out_features=d_data * (poly_dim+1)**2,device=device)
        #self.b_net.weight = nn.Parameter(torch.zeros(self.b_net.weight.shape,device=device),requires_grad=False)
        #print(self.b_net.weight)
        #torch.nn.init.uniform_(self.b_net.weight,a=-1/tau**2,b=1/tau**2)
        #print(self.b_net.weight)
        #torch.nn.init.uniform_(self.b_net.bias,a=-1/tau**2,b=1/tau**2)
        # d data above: since we only parameterize  d2y/dt2
        self.omega_net = nn.Linear(in_features=d_control,out_features=d_data,device=device) #unconstrained
        self.gamma_net = nn.Linear(in_features=d_control,out_features=d_data,device=device) # output unconstrained, but weights nonneg
        self.b_net = nn.Linear(in_features=d_control,out_features=d_data,device=device) # weights non-neg
        self.b = nn.Parameter(torch.zeros(1),requires_grad=True).to(device)
        #torch.nn.init.uniform_(self.b,a=0,b=2/tau)
        self.d_net = nn.Linear(in_features=d_control,out_features=d_data,device=device)
        torch.nn.init.uniform_(self.omega_net.weight,a=-omega_scale/tau,b=omega_scale/tau)
        torch.nn.init.uniform_(self.omega_net.bias,a=-omega_scale/tau,b=omega_scale/tau)
        torch.nn.init.uniform_(self.gamma_net.weight,a=0,b=2/tau)
        torch.nn.init.uniform_(self.gamma_net.bias,a=0,b=2/tau)
        torch.nn.init.uniform_(self.b_net.weight,a=-1/tau,b=1/tau)
        torch.nn.init.uniform_(self.b_net.bias,a=-1/tau,b=1/tau)
        torch.nn.init.uniform_(self.d_net.weight,a=-1/tau**2,b=1/tau**2)
        torch.nn.init.uniform_(self.d_net.bias,a=-1/tau**2,b=1/tau**2)

        self.tau = tau
        self.omega=omega
        if noInput:
            print("running with no input")
        if good_init:
            b_params = torch.zeros(self.b_net.weight.shape,device=device)

            omega_params = torch.zeros(self.omega_net.weight.shape,device=device)
            b_biases = torch.zeros(self.b_net.bias.shape,device=device)
            omega_biases = torch.zeros(self.omega_net.bias.shape,device=device)
            omega_biases[0] = 2*np.pi*self.omega/self.tau
            self.b_net.weight = nn.Parameter(b_params)
            self.b_net.bias = nn.Parameter(b_biases)
            self.omega_net.weight = nn.Parameter(omega_params)
            self.omega_net.bias = nn.Parameter(omega_biases)

            
        self.noInput = noInput
        self.poly_dim = poly_dim+1
        self.d_data = d_data 
        self.powers = torch.arange(0,poly_dim+1,device=device)

    def _clip_weights(self):

        self.gamma_net.apply(self.clipper)
        #self.b.apply(self.clipper)
        #self.b = self.b.clamp(min=0)
        #self.b_net.apply(self.clipper)
        

    def forward(self,x,dt,idx):
        """
        predicts second derivative at time t.
        all other predictions should be done in the train look (train_utils.py)
        """

        B,L,d = x.shape
        # x: x_0, x_dt, x_2dt,...

        xdot= deriv_approx_dy(x) # this gives: dxdt (currently unitless)
        
        # dx: dx_4dt,dx_5dt,dx_6dt,..., dx_(L-4)dt
        z = torch.cat([x[:,4:-4,:],xdot],dim=-1)
        L = z.shape[1]
        x_in = torch.cat([z, torch.flip(z,[1])],dim=-1)
        
        states = self.controlMamba(self.control_proj(x_in))
        states = torch.flip(states,[1])
        
        b = nn.ReLU()(self.b_net(states))/self.tau #.view(B,L,d,self.poly_dim,self.poly_dim) # B x L x 2d x P x P
        d = nn.ReLU()(self.d_net(states))
        if self.noInput:
            d = torch.zeros(d.shape,device='cuda')
        omega = self.omega_net(states)
        gamma = self.gamma_net(states)/self.tau
        
        #print(omega[0,:40,...])
        #print(omega[0,:40,0],2*np.pi*self.omega)
        #b[:,:,:,1,0] = -(omega**2)
        #b[:,:,:,0,1] = gamma
        """
        if self.noInput:
            b[:,:,:,0,0] = 0
        if idx %100 == 0:
            ax = plt.gca()
            ax.plot(omega[0,:40,0].detach().cpu().numpy()*model.tau,label=r'$\omega(t)$')
            ax.plot(gamma[0,:40,0].detach().cpu().numpy()*model.tau,label=r'$\gamma(t)$')
            ax.set_ylabel("omega,gamma")
            ax.legend()
            ax2=ax.twinx()
            ax2.plot(b[0,:40,0,1,1].detach().cpu().numpy()*model.tau,label=r'$g(y,\dot{y},t)$',color='tab:green')
            ax2.set_ylabel("g()")
            ax2.legend()
            plt.show()
            plt.close()
            ax = plt.gca()
            ax.plot(torch.diff(omega,dim=1)[0,:40,0].detach().cpu().numpy()*model.tau,label=r'$\omega(t)$ diff')
            ax.legend()
            ax2=ax.twinx()
            ax2.plot(torch.diff(b,dim=1)[0,:40,0,1,1].detach().cpu().numpy()*model.tau,label=r'$g(y,\dot{y},t)$ diff',color='tab:green')
            ax2.set_ylabel("g()")
            ax2.legend()
            plt.show()
            plt.close()
            #assert False
        """
        #b*= self.tau**2
        z[:,:,-1] /= dt
        z1 = z[:,:,:1]
        power_mat_z1 = z[:,:,:1]**2 #z[:,:,:1,None].expand(-1,-1,-1,self.poly_dim) # B x L x d -> B x L x 2d x P
        z2 = z[:,:,1:]#,None].expand(-1,-1,-1,self.poly_dim) # B x L x d -> B x L x 2d x P
        #power_mat_z1 = power_mat_z1.pow(self.powers)
        #power_mat_z2 = power_mat_z2.pow(self.powers)
        #pow1 = torch.einsum('bldjk,bldj->bldk',b,power_mat_z1)
        yhat = -(omega**2)*z1 + gamma * z2 - b*power_mat_z1 * z2 - d #torch.einsum('bldk,bldk->bld',pow1,power_mat_z2)
        #print(torch.amax(power_mat_z1))
        #print(torch.amax(power_mat_z2))

        #db = torch.diff(b,dim=1)
        smooth_penalty = self.smooth_penalty(torch.diff(omega,dim=1)[:,1:,:]).mean() + self.smooth_penalty(torch.diff(gamma,dim=1)[:,1:,:]).mean() + self.smooth_penalty(torch.diff(b,dim=1)[:,1:,:]).mean()#(torch.diff(omega,dim=1)[:,1:,:]**2).mean() + (torch.diff(gamma,dim=1)[:,1:,:]**2).mean() +\
                        #(torch.diff(b,dim=1)[:,1:,:]**2).mean()
        #                     (torch.diff(d,dim=1)**2).sum(dim=(-1,-2)).mean()        #  +       
        #(db[:,:,:,2:self.poly_dim,:]**2).sum(dim=(-1,-2,-3)).mean() + (db[:,:,:,:,2:self.poly_dim]**2).sum(dim=(-1,-2,-3)).mean() +\
                        #(db[:,:,:,1,1]**2).sum(dim=(-1,-2)).mean()
        size_penalty =  (d).abs().mean()#b.pow(2).sum(dim=(-1,-2)).mean() #*self.tau**2).abs().pow(2)#.sum(dim=(-1,-2)).mean()
        #b[:,:,:,2:self.poly_dim,:].abs().pow(2).sum(dim=(-1,-2,-3)).mean() + b[:,:,:,:,2:self.poly_dim].abs().pow(2).sum(dim=(-1,-2,-3)).mean() +\
        #b[:,:,:,1,1].abs().pow(2).sum(dim=(-1,-2)).mean()
        #print(size_penalty)
        return yhat,states,size_penalty + smooth_penalty/3

    def get_funcs(self,x,dt):

        B,_,d = x.shape
        # x: x_0, x_dt, x_2dt,...
    
        xdot= deriv_approx_dy(x)
        # dx: dx_4dt,dx_5dt,dx_6dt,..., dx_(l-4)dt
    
        z = torch.cat([x[:,4:-4,:],xdot],dim=-1)
        L = z.shape[1]
        x_in = torch.cat([z, torch.flip(z,[1])],dim=-1)
        states = self.controlMamba(self.control_proj(x_in))
        states = torch.flip(states,[1])
        
        #nn.ReLU()(self.b_net(states))#.view(B,L,d,self.poly_dim,self.poly_dim) # B x L x 2d x P x P
        d = nn.ReLU()(self.d_net(states))
        if self.noInput:
            d = torch.zeros(d.shape,device='cuda')
        omega = self.omega_net(states)
        gamma = self.gamma_net(states)
        
        omega *= self.tau
        gamma *= self.tau
        d *= self.tau**2
    
        z /= dt
        z1 = z[:,:,:1]
        z2 = z[:,:,1:]
        power_mat_z1 = z[:,:,:1].pow(2) #expand(-1,-1,-1,model.poly_dim) # B x 2d -> B x 2d x P

        b = nn.ReLU()(self.b_net(states)) #self.b * torch.ones(power_mat_z1.shape,device='cuda')
        b *= self.tau
        b_out =b * power_mat_z1 * z2
        
    
        
        return omega,gamma,b,b_out,d,states
        

    def integrate(self,x,dt):

        B,_,d = x.shape
        # x: x_0, x_dt, x_2dt,...

        xdot= deriv_approx_dy(x)
        # dx: dx_4dt,dx_5dt,dx_6dt,..., dx_(l-4)dt
        #L = xdot.shape[1]
        z = torch.cat([x[:,4:-4,:],xdot],dim=-1)
        L = z.shape[1]
        x_in = torch.cat([z, torch.flip(z,[1])],dim=-1)
        z0 = z[:,0,:]
        z0[:,-1] /= dt
        
        states = self.controlMamba(self.control_proj(x_in))
        states = torch.flip(states,[1])

        b = nn.ReLU()(self.b_net(states))#.view(B,L,d,self.poly_dim,self.poly_dim) # B x L x 2d x P x P
        d = nn.ReLU()(self.d_net(states))
        if self.noInput:
            d = torch.zeros(d.shape,device='cuda')
        omega = self.omega_net(states)
        gamma = nn.ReLU()(self.gamma_net(states))
        #if self.noInput:
        #    b[:,:,:,0,0] = 0
        
        #b[:,:,:,1,0] = -(omega**2) # this means we multiply here by tau**2
        b *= self.tau #/dt**2
        b = b.detach().cpu().numpy()
        pows = self.powers.detach().cpu().numpy()
        def dz(t,z):

            # t: time, should have a timestep of roughly 1. treat as ZOH
            # z: B x 2d
            b_ind = int(t)
            b_step = self.b * self.tau**2 #b[0,b_ind,:,:,:]
            omega_step = omega[:,b_ind,:]
            gamma_step = gamma[:,b_ind,:]
            d_step = d[:,b_ind,:]
            
            #power_mat_z1 = np.tile(z[:1,None],(1,self.poly_dim)) # B x 2d -> B x 2d x P
            #power_mat_z2 = np.tile(z[1:,None],(1,self.poly_dim)) # B x 2d -> B x 2d x P
            #power_mat_z1 = np.power(power_mat_z1,pows)
            #power_mat_z2 = np.power(power_mat_z2,pows)
            z1 = z[:1]
            power_mat_z1 = z[:1]**2 #z[:,:,:1,None].expand(-1,-1,-1,self.poly_dim) # B x L x d -> B x L x 2d x P
            z2 = z[1:]
            #pow1 = np.einsum('djk,dj->dk',b_step,power_mat_z1)
            dz2 = -(omega_step**2)*z1 + gamma_step * z2 - b_step*power_mat_z1 * z2 - d_step #np.einsum('dk,dk->d',pow1,power_mat_z2)
            
            dz1 = z[1]
            
            return np.hstack([dz1,dz2])
        t_steps = np.arange(0,L*dt + dt/2,dt)[:L]
        obj = solve_ivp(dz,(0,L*dt),z0.squeeze().detach().cpu().numpy(),t_eval=t_steps,method='Radau')

        return obj.y
    
class kernel_ouroboros(ouroboros):


    def __init__(self,d_data,
                 d_control,
                 d_out=1,
                 n_layers_data=2,
                 n_layers_control=2,
                 d_state_data=16,
                 d_state_control=16,
                 d_conv_data=4,
                 d_conv_control=4,
                 expand_factor_data=1,
                 expand_factor_control=1,
                 device='cuda',
                 n_kernels=2,
                 tau = 1000,
                 omega=4000,
                 noInput = False):
        
        super(constrained_ouroboros,self).__init__(d_data,\
                 d_control,\
                 d_out,\
                 n_layers_data,\
                 n_layers_control,\
                 d_state_data,\
                 d_state_control,\
                 d_conv_data,\
                 d_conv_control,\
                 expand_factor_data,\
                 expand_factor_control,\
                 device)
        
        self.weights = nn.Linear(in_features=d_control,out_features=n_kernels,device=device)
        self.centers = nn.Linear(in_features=d_control,out_features=2*n_kernels,device=device)
        self.variances = nn.Linear(in_features=d_control,out_features=2*n_kernels,device=device)
        #print(self.b_net.weight)
        torch.nn.init.uniform_(self.b_net.weight,a=-1/tau,b=1/tau)
        #print(self.b_net.weight)
        torch.nn.init.uniform_(self.b_net.bias,a=-1/tau,b=1/tau)
        # d data above: since we only parameterize  d2y/dt2
        self.omega_net = nn.Linear(in_features=d_control,out_features=d_data,device=device)
        self.gamma_net = nn.Linear(in_features=d_control,out_features=d_data,device=device)
        
        self.tau = tau
        self.omega=omega
            
        self.noInput = noInput
        self.n_kernels = n_kernels
        self.d_data = d_data 
        #self.powers = torch.arange(0,poly_dim,device=device)

    def _apply_kernels(self,x,centers,variances):

        ### assumes x is already stacked [y,ydot]

        pass



    def forward(self,x,dt):
        """
        predicts second derivative at time t.
        all other predictions should be done in the train look (train_utils.py)
        """

        B,L,d = x.shape
        # x: x_0, x_dt, x_2dt,...

        xdot= deriv_approx_dy(x) # this gives: dx
        
        # dx: dx_4dt,dx_5dt,dx_6dt,..., dx_(L-4)dt
        z = torch.cat([x[:,4:-4,:],xdot],dim=-1)
        L = z.shape[1]
        x_in = torch.cat([z, torch.flip(z,[1])],dim=-1)
        
        states = self.controlMamba(self.control_proj(x_in))
        states = torch.flip(states,[1])
        
        b = self.b_net(states).view(B,L,d,self.poly_dim,self.poly_dim) # B x L x 2d x P x P
        
        omega = self.omega_net(states)
        #print(omega[0,:40,...])
        #print(omega[0,:40,0],2*np.pi*self.omega)
        b[:,:,0,1,0] = -(omega**2).squeeze()
        if self.noInput:
            b[:,:,:,0,0] = 0

        #b*= self.tau**2
        z[:,:,-1] /= dt
        power_mat_z1 = z[:,:,:1,None].expand(-1,-1,-1,self.poly_dim) # B x L x d -> B x L x 2d x P
        power_mat_z2 = z[:,:,1:,None].expand(-1,-1,-1,self.poly_dim) # B x L x d -> B x L x 2d x P
        power_mat_z1 = power_mat_z1.pow(self.powers)
        power_mat_z2 = power_mat_z2.pow(self.powers)
        pow1 = torch.einsum('bldjk,bldj->bldk',b,power_mat_z1)
        yhat = torch.einsum('bldk,bldk->bld',pow1,power_mat_z2)
        #print(torch.amax(power_mat_z1))
        #print(torch.amax(power_mat_z2))

        return yhat,states

    def get_funcs(self,x,dt):

        B,_,d = x.shape
        # x: x_0, x_dt, x_2dt,...

        xdot= deriv_approx_dy(x)
        # dx: dx_4dt,dx_5dt,dx_6dt,..., dx_(l-4)dt

        z = torch.cat([x[:,4:-4,:],xdot],dim=-1)
        L = z.shape[1]
        x_in = torch.cat([z, torch.flip(z,[1])],dim=-1)
        states = self.controlMamba(self.control_proj(x_in))
        states = torch.flip(states,[1])
        
        b = self.b_net(states).view(B,L,d,self.poly_dim,self.poly_dim) # B x L x 2d x P x P
        omega = self.omega_net(states)
        
        #b[:,:,0,1,0] = -(omega**2).squeeze() 
        b *= self.tau**2

        B,L,D,P1,P2 = b.shape
        A = torch.cat([torch.cat([torch.zeros((B,L,D,1,1),device='cuda'),torch.ones((B,L,D,1,1),device='cuda')/self.tau**2],dim=-1),\
           torch.stack([b[:,:,:,1,0].detach().clone(),b[:,:,:,0,1].detach().clone()],dim=-1)[:,:,:,None,:]],dim=-2)

        assert A.shape[-1] == 2, print(A.shape)
        assert A.shape[-2] == 2,print(A.shape) 

        c = b[:,:,:,0,0].detach().clone()

        b[:,:,:,0,0] = 0
        z[:,:,-1] /= dt
        power_mat_z1 = z[:,:,:1,None].expand(-1,-1,-1,self.poly_dim) # B x 2d -> B x 2d x P
        power_mat_z2 = z[:,:,1:,None].expand(-1,-1,-1,self.poly_dim) # B x 2d -> B x 2d x P
        power_mat_z1 = power_mat_z1.pow(self.powers)
        power_mat_z2 = power_mat_z2.pow(self.powers)
        
        pow1 = torch.einsum('bldjk,bldj->bldk',b,power_mat_z1)
        
        b_out = torch.einsum('bldk,bldk->bld',pow1,power_mat_z2)

        
        return A,b,b_out,c,states
    

    def integrate(self,x,dt):

        B,_,d = x.shape
        # x: x_0, x_dt, x_2dt,...

        xdot= deriv_approx_dy(x)
        # dx: dx_4dt,dx_5dt,dx_6dt,..., dx_(l-4)dt
        #L = xdot.shape[1]
        z = torch.cat([x[:,4:-4,:],xdot],dim=-1)
        L = z.shape[1]
        x_in = torch.cat([z, torch.flip(z,[1])],dim=-1)
        z0 = z[:,0,:]
        z0[:,-1] /= dt
        
        states = self.controlMamba(self.control_proj(x_in))
        states = torch.flip(states,[1])

        b = self.b_net(states).view(B,L,d,self.poly_dim,self.poly_dim) # B x L x 2d x P x P
        omega = self.omega_net(states)#*self.tau
        if self.noInput:
            b[:,:,:,0,0] = 0
        
        b[:,:,:,1,0] = -(omega**2) # this means we multiply here by tau**2
        b *= self.tau**2 #/dt**2
        b = b.detach().cpu().numpy()
        pows = self.powers.detach().cpu().numpy()
        def dz(t,z):

            # t: time, should have a timestep of roughly 1. treat as ZOH
            # z: B x 2d
            b_ind = int(t)
            b_step = b[0,b_ind,:,:,:]
            
            power_mat_z1 = np.tile(z[:1,None],(1,self.poly_dim)) # B x 2d -> B x 2d x P
            power_mat_z2 = np.tile(z[1:,None],(1,self.poly_dim)) # B x 2d -> B x 2d x P
            power_mat_z1 = np.power(power_mat_z1,pows)
            power_mat_z2 = np.power(power_mat_z2,pows)
            
            pow1 = np.einsum('djk,dj->dk',b_step,power_mat_z1)
            dz2 = np.einsum('dk,dk->d',pow1,power_mat_z2)
            
            dz1 = z[1]
            
            return np.hstack([dz1,dz2])
        t_steps = np.arange(0,L*dt + dt/2,dt)[:L]
        obj = solve_ivp(dz,(0,L*dt),z0.squeeze().detach().cpu().numpy(),t_eval=t_steps,method='Radau')

        return obj.y





        


        

        

        


        

        
