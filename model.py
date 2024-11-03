from mambapy.mamba import Mamba, MambaConfig

import torch
import torch.nn as nn
from torchaudio.functional import lowpass_biquad
from torchdiffeq import odeint
import numpy as np
from scipy.integrate import solve_ivp


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
                 data_fs=41100,
                 smooth_length=0.007,
                good_init=True,
                tau = 1000):
        
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
        
        self.b_net = nn.Linear(in_features=d_control-1,out_features=2*d_data * poly_dim**2,device=device)
        self.omega_net = nn.Linear(in_features=d_control-1,out_features=1,device=device)
        self.tau = tau
        if good_init:
            b_params = torch.zeros(self.b_net.weight.shape,device=device)
            #print(b_params.shape)
            #print(self.b_net.weight.shape)
            omega_params = torch.zeros(self.omega_net.weight.shape,device=device)
            b_biases = torch.zeros(self.b_net.bias.shape,device=device)
            #b_biases[1] = 1
            omega_biases = torch.zeros(self.omega_net.bias.shape,device=device)
            omega_biases[0] = 2*np.pi*2000/self.tau
            self.b_net.weight = nn.Parameter(b_params)
            self.b_net.bias = nn.Parameter(b_biases)
            self.omega_net.weight = nn.Parameter(omega_params)
            self.omega_net.bias = nn.Parameter(omega_biases)

            
        
        self.poly_dim = poly_dim
        self.d_data = d_data 
        self.powers = torch.arange(0,poly_dim,device=device)


    def forward(self,x,y):

        B,L,d = x.shape
        xdot= y - x
        z = torch.cat([x,xdot],dim=-1)
        #power_mat = z[:,:,:,None].expand(-1,-1,-1,self.poly_dim) # B x L x 2d -> B x L x 2d x P
        #power_mat = power_mat.pow(self.powers)
        
        # CHANGE: double flippy
        x_in = torch.cat([xdot,y],dim=-1)
        x_in = torch.cat([x_in, torch.flip(x_in,[1])])
        state_pred = self.controlMamba(self.control_proj(x_in))
        state_pred = torch.flip(state_pred,[1])#.transpose(-1,-2) #torch.flip(torch.nn.SiLU()(state_pred),[1])
        init_conditions = state_pred[:,0,0]
        states = state_pred[:,:,1:]
        #print(states.shape)
        #print(self.b_net.weight.shape)
        b = self.b_net(states).view(B,L,2*d,self.poly_dim,self.poly_dim) # B x L x 2d x P x P
        omega = self.omega_net(states)*self.tau
        b *= self.tau
        b[:,:,0,1,0] = 0
        b[:,:,1,0,1] = 1
        b[:,:,1,1,0] = -(omega**2).squeeze()
        #b *= self.tau**2
        penalty = ((b[:,:,:,2:,:2]**2).sum(dim=(-1,-2)) + (b[:,:,:,:2,2:]**2).sum(dim=(-1,-2)) + (b[:,:,:,2:,2:]**2).sum(dim=(-1,-2))).sqrt().mean()
        
        power_mat_z1 = z[:,:,:1,None].expand(-1,-1,-1,self.poly_dim) # B x L x 2d -> B x L x 2d x P
        power_mat_z2 = z[:,:,1:,None].expand(-1,-1,-1,self.poly_dim) # B x L x 2d -> B x L x 2d x P
        power_mat_z1 = power_mat_z1.pow(self.powers)
        power_mat_z2 = power_mat_z2.pow(self.powers)
        power_mat_z1 = power_mat_z1.expand(-1,-1,2,-1)
        power_mat_z2 = power_mat_z2.expand(-1,-1,2,-1)
        pow1 = torch.einsum('bldjk,bldj->bldk',b,power_mat_z1)
        #print(pow1)
        #print(pow1.shape)
        #print(pow1.shape)
        #print(b.shape)
        #print(power_mat_z2.shape)
        
        yhat = torch.einsum('bldk,bldk->bld',pow1,power_mat_z2)

        #x2 = init_conditions[:,None] + torch.cumsum(zdot[:,:,0],dim=1)

        #yhat = zdot #torch.cat([x2[:,:,None],zdot],dim=-1)
        return yhat[:,1:,:],states#,penalty

    def get_funcs(self,x,y):

        B,L,d = x.shape
        xdot= y - x
        z = torch.cat([x,xdot],dim=-1)
        #power_mat = z[:,:,:,None].expand(-1,-1,-1,self.poly_dim) # B x L x 2d -> B x L x 2d x P
        #power_mat = power_mat.pow(self.powers)
        

        state_pred = self.controlMamba(self.control_proj(torch.flip(torch.cat([xdot,y],dim=-1),[1])))
        state_pred = torch.flip(state_pred,[1])#.transpose(-1,-2) #torch.flip(torch.nn.SiLU()(state_pred),[1])
        init_conditions = state_pred[:,:,0]
        states = state_pred[:,:,1:]
        b = self.b_net(states).view(B,L,2*d,self.poly_dim,self.poly_dim) # B x L x 2d x P x P
        omega = self.omega_net(states)*self.tau
        b *= self.tau
        b[:,:,0,1,0] = 0
        b[:,:,1,0,1] = 1
        b[:,:,1,1,0] = -(omega**2).squeeze()
        #b *= self.tau**2
        penalty = ((b[:,:,:,2:,:2]**2).sum(dim=(-1,-2)) + (b[:,:,:,:2,2:]**2).sum(dim=(-1,-2)) + \
                   (b[:,:,:,2:,2:]**2).sum(dim=(-1,-2))).sqrt().mean()
        #print(penalty)
        A = torch.stack([b[:,:,:,1,0].detach().clone(),torch.flip(b,[2])[:,:,:,0,1].detach().clone()],dim=-1) #b[:,:,:,[0,1],[0,1]]
        #print(A.shape)
        #print(A[:0,0,:,:])

        c = b[:,:,:,0,0].detach().clone()

        b[:,:,:,0,0] = 0
        #b[:,:,:,1,0] = 0
        #b[:,:,:,0,1] = 0
        
        power_mat_z1 = z[:,:1,:,None].expand(-1,-1,-1,self.poly_dim) # B x 2d -> B x 2d x P
        power_mat_z2 = z[:,1:,:,None].expand(-1,-1,-1,self.poly_dim) # B x 2d -> B x 2d x P
        power_mat_z1 = power_mat_z1.pow(self.powers)
        power_mat_z2 = power_mat_z2.pow(self.powers)
        power_mat_z1 = power_mat_z1.expand(-1,-1,2,-1)
        power_mat_z2 = power_mat_z2.expand(-1,-1,2,-1)
        pow1 = torch.einsum('bldjk,bldj->bldk',b,power_mat_z1)
        #print(pow1)
        #print(pow1.shape)
        
        b_out = torch.einsum('bldk,bldk->bld',pow1,power_mat_z2)
        #b_out = pow2#torch.einsum('bldj,bldj->bld',pow1,pow2)

        
        return A,b,b_out,c,state_pred
    

    def integrate(self,data):

        x,y = data[:,:-1,:],data[:,1:,:]
        B,L,d = x.shape
        dx = y - x 
        z = torch.cat([x,dx],dim=-1)
        z0 = z[:,0,:]


        state_pred = self.controlMamba(self.control_proj(torch.flip(torch.cat([dx,y],dim=-1),[1])))
        state_pred = torch.flip(state_pred,[1])#.transpose(-1,-2) #torch.flip(torch.nn.SiLU()(state_pred),[1])
        #init_conditions = state_pred[:,:,0]
        states = state_pred[:,:,1:]
        b = self.b_net(states).view(B,L,2*d,self.poly_dim,self.poly_dim) # B x L x 2d x P x P
        omega = self.omega_net(states)*self.tau
        b *= self.tau
        b[:,:,0,1,0] = 0
        b[:,:,1,0,1] = 1
        b[:,:,1,1,0] = -(omega**2).squeeze()
        b = b.detach().cpu().numpy()
        #b *= self.tau**2
        def dz(t,z):

            # t: time, should have a timestep of roughly 1. treat as ZOH
            # z: B x 2d
            #print(z)
            b_ind = int(t)#.type(torch.IntTensor)
            #print(b_ind)
            b_step = b[:,b_ind,:,:,:]
            #print(b_step.shape)
            power_mat_z1 = np.tile(z[:,:1,None],(1,1,self.poly_dim)) # B x 2d -> B x 2d x P
            power_mat_z2 = np.tile(z[:,1:,None],(1,1,self.poly_dim)) # B x 2d -> B x 2d x P
            power_mat_z1 = np.power(power_mat_z1,self.powers)
            power_mat_z2 = np.power(power_mat_z2,self.powers)
            power_mat_z1 = np.tile(power_mat_z1,(1,2,1))
            power_mat_z2 = np.tile(power_mat_z2,(1,2,1))
            pow1 = np.einsum('bdjk,bdj->bdk',b_step,power_mat_z1)
            #print(pow1)
            #print(pow1.shape)
            
            dz = np.einsum('bdk,bdk->bd',pow1,power_mat_z2)
            #print(dz)
            
            return dz
        #print(L)
        t_steps = np.arange(0,L)#.type(torch.FloatTensor)
        #print({'step_size':0.5})
        obj = solve_ivp(dz,(0,L),z0.squeeze().detach().cpu().numpy(),t_eval=t_steps)
        #z_int = odeint(dz,,t_steps,method='rk4')#,options=dict(step_size:0.5))

        return obj.y

        


        

        

        


        

        
