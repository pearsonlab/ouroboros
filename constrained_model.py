import torch
from torch import nn
from mambapy.mamba import Mamba, MambaConfig
from model_utils import smooth, NonNegClipper
from utils import deriv_approx_dy

from scipy.integrate import solve_ivp
import numpy as np
import time

class constrained_ouroboros(nn.Module):



    def __init__(self,d_data,
                 d_out=1,
                 n_layers=2,
                 d_state=16,
                 d_conv=4,
                 expand_factor=1,
                 device='cuda',
                tau = 1000,
                noInput = False,
                omega_scale = 10000,
                 smooth_len=0.001,
                smooth_penalty=lambda x: torch.var(x,dim=1)):
        
        super(constrained_ouroboros,self).__init__()

        omegaConfig = MambaConfig(d_model=4*d_data,\
                                    n_layers=n_layers,d_state=d_state,\
                                    d_conv=d_conv,expand_factor=expand_factor)
        gammaConfig = MambaConfig(d_model=4*d_data,\
                                    n_layers=n_layers,d_state=d_state,\
                                    d_conv=d_conv,expand_factor=expand_factor)
        dConfig = MambaConfig(d_model=4*d_data,\
                                    n_layers=n_layers,d_state=d_state,\
                                    d_conv=d_conv,expand_factor=expand_factor)

        self.omegaMamba = Mamba(omegaConfig).to(device)
        self.gammaMamba = Mamba(gammaConfig).to(device)
        self.dMamba = Mamba(dConfig).to(device)
        self.omega_net = nn.Linear(in_features=4*d_data,out_features=d_data,device=device) #unconstrained
        self.gamma_net = nn.Linear(in_features=4*d_data,out_features=d_data,device=device) # output unconstrained, but weights nonneg
        self.b = nn.Parameter(torch.zeros(1),requires_grad=True).to(device) #non-neg
        self.d_net = nn.Linear(in_features=4*d_data,out_features=d_data,device=device) #non-neg
        #torch.nn.init.uniform_(self.omega_net.weight,a=-omega_scale/tau,b=omega_scale/tau)
        #torch.nn.init.uniform_(self.omega_net.bias,a=-omega_scale/tau,b=omega_scale/tau)
        #torch.nn.init.uniform_(self.gamma_net.weight,a=0,b=2/tau)
        #torch.nn.init.uniform_(self.gamma_net.bias,a=0,b=2/tau)
        #torch.nn.init.uniform_(self.b,a=0,b=2/tau)
        #torch.nn.init.uniform_(self.d_net.weight,a=-1/tau**2,b=1/tau**2)
        #torch.nn.init.uniform_(self.d_net.bias,a=-1/tau**2,b=1/tau**2)
        
        
        self.clipper = NonNegClipper(min=0)
        self.smooth_penalty = smooth_penalty
        self.smooth_len=smooth_len

        self.tau = tau
        #self.omega=omega
        if noInput:
            print("running with no input")
            
        self.noInput = noInput
        self.d_data = d_data 

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
        smooth_len = int(round(self.smooth_len/dt))

        xdot= deriv_approx_dy(x) # this gives: dxdt (currently unitless)
        
        # dx: dx_4dt,dx_5dt,dx_6dt,..., dx_(L-4)dt
        z = torch.cat([x[:,4:-4,:],xdot],dim=-1)
        L = z.shape[1]
        x_in = torch.cat([z, torch.flip(z,[1])],dim=-1)
        
        omegaControl = self.omegaMamba(x_in)
        gammaControl = self.gammaMamba(x_in)
        dControl = self.dMamba(x_in)

        dControl=smooth(dControl.abs(),smooth_len)
        omegaControl=smooth(omegaControl.abs(),smooth_len)
        gammaControl=smooth(gammaControl,smooth_len)
        b = nn.ReLU()(self.b)/self.tau #.view(B,L,d,self.poly_dim,self.poly_dim) # B x L x 2d x P x P
        d = nn.ReLU()(self.d_net(dControl))
        if self.noInput:
            d = torch.zeros(d.shape,device='cuda')
        omega = self.omega_net(omegaControl)
        gamma = self.gamma_net(gammaControl)/self.tau
        
       
        z[:,:,-1] /= dt
        z1 = z[:,:,:1]
        power_mat_z1 = z[:,:,:1]**2 
        z2 = z[:,:,1:]

        yhat = -(omega**2)*z1 + gamma * z2 - b*power_mat_z1 * z2 - d

        smooth_penalty = self.smooth_penalty(torch.diff(omega,dim=1)[:,1:,:]).mean() + self.smooth_penalty(torch.diff(gamma,dim=1)[:,1:,:]).mean() + self.smooth_penalty(torch.diff(d,dim=1)[:,1:,:]).mean()
        size_penalty = 0 # (d).abs().mean()
        return yhat,torch.cat([omegaControl,gammaControl,dControl],dim=-1),size_penalty + smooth_penalty/3

    def get_funcs(self,x,dt):

        B,_,d = x.shape
        # x: x_0, x_dt, x_2dt,...
        smooth_len = int(round(self.smooth_len/dt))
        xdot= deriv_approx_dy(x)
        # dx: dx_4dt,dx_5dt,dx_6dt,..., dx_(l-4)dt
    
        z = torch.cat([x[:,4:-4,:],xdot],dim=-1)
        L = z.shape[1]
        x_in = torch.cat([z, torch.flip(z,[1])],dim=-1)
        omegaControl = self.omegaMamba(x_in)
        gammaControl = self.gammaMamba(x_in)
        dControl = self.dMamba(x_in)
        dControl=smooth(dControl.abs(),smooth_len)
        omegaControl=smooth(omegaControl.abs(),smooth_len)
        gammaControl=smooth(gammaControl,smooth_len)
        b = nn.ReLU()(self.b)/self.tau #.view(B,L,d,self.poly_dim,self.poly_dim) # B x L x 2d x P x P
        d = nn.ReLU()(self.d_net(dControl))
        if self.noInput:
            d = torch.zeros(d.shape,device='cuda')
        omega = self.omega_net(omegaControl)
        gamma = self.gamma_net(gammaControl)/self.tau
        
        omega *= self.tau
        gamma *= self.tau
        d *= self.tau**2
    
        z /= dt
        z1 = z[:,:,:1]
        z2 = z[:,:,1:]
        power_mat_z1 = z[:,:,:1].pow(2) #expand(-1,-1,-1,model.poly_dim) # B x 2d -> B x 2d x P

        #b = nn.ReLU()(self.b_net(states)) #self.b * torch.ones(power_mat_z1.shape,device='cuda')
        b *= self.tau
        b_out =b * power_mat_z1 * z2
        
    
        
        return omega,gamma,b,b_out,d,torch.cat([omegaControl,gammaControl,dControl],dim=-1)
    
    def integrate(self,x,dt):

        B,_,d = x.shape
        # x: x_0, x_dt, x_2dt,...
        smooth_len = int(round(self.smooth_len/dt))
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

        omegaControl = self.omegaMamba(x_in)
        gammaControl = self.gammaMamba(x_in)
        dControl = self.dMamba(x_in)
        dControl=smooth(dControl.abs(),smooth_len)
        omegaControl=smooth(omegaControl.abs(),smooth_len)
        gammaControl=smooth(gammaControl,smooth_len)
        b = nn.ReLU()(self.b)/self.tau #.view(B,L,d,self.poly_dim,self.poly_dim) # B x L x 2d x P x P
        d = nn.ReLU()(self.d_net(dControl))
        if self.noInput:
            d = torch.zeros(d.shape,device='cuda')
        omega = self.omega_net(omegaControl)
        gamma = self.gamma_net(gammaControl)/self.tau
        
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
    
    

class arneodobouros(nn.Module):


    def __init__(self,d_data,
                 n_layers=2,
                 d_state=16,
                 d_conv=4,
                 expand_factor=1,
                 device='cuda',
                 tau = 1000,
                 noInput = False,
                 omega_scale = 10000,
                 smooth_len=0.001,
                smooth_penalty=lambda x: torch.var(x,dim=1)):
        
        super(arneodobouros,self).__init__()

        self.device=device
        alphaConfig = MambaConfig(d_model=4*d_data,\
                                    n_layers=n_layers,d_state=d_state,\
                                    d_conv=d_conv,expand_factor=expand_factor)
        betaConfig = MambaConfig(d_model=4*d_data,\
                                    n_layers=n_layers,d_state=d_state,\
                                    d_conv=d_conv,expand_factor=expand_factor)
        dConfig = MambaConfig(d_model=4*d_data,\
                                    n_layers=n_layers,d_state=d_state,\
                                    d_conv=d_conv,expand_factor=expand_factor)
        
        self.alphaMamba = Mamba(alphaConfig).to(device)
        self.betaMamba = Mamba(betaConfig).to(device)
        self.dMamba = Mamba(dConfig).to(device)

        self.alpha_net = nn.Linear(in_features=4*d_data,out_features=d_data,device=device) #unconstrained
        self.beta_net = nn.Linear(in_features=4*d_data,out_features=d_data,device=device) # output unconstrained, but weights nonneg
        #self.b = nn.Parameter(torch.zeros(1),requires_grad=True).to(device) #non-neg
        self.d_net = nn.Linear(in_features=4*d_data,out_features=d_data,device=device) #non-neg

        tau = nn.Parameter(torch.tensor([tau],dtype=torch.float64,device=device), requires_grad=True)
        self.tau = tau
        self.smooth_len = smooth_len
        self.noInput = noInput

    def forward(self,x,dt):


        B,L,d = x.shape
        # x: x_0, x_dt, x_2dt,...
        smooth_len = int(round(self.smooth_len/dt))

        xdot= deriv_approx_dy(x) # this gives: dxdt (currently unitless)
        
        # dx: dx_4dt,dx_5dt,dx_6dt,..., dx_(L-4)dt
        z = torch.cat([x[:,4:-4,:],xdot],dim=-1)
        L = z.shape[1]
        x_in = torch.cat([z, torch.flip(z,[1])],dim=-1)
        
        alphaControl = self.alphaMamba(x_in)
        betaControl = self.betaMamba(x_in)
        dControl = self.dMamba(x_in)

        dControl,betaControl = smooth(dControl.abs(),smooth_len),smooth(betaControl.abs(),smooth_len)

        d = nn.ReLU()(self.d_net(dControl))
        if self.noInput:
            d = torch.zeros(d.shape,device='cuda')
        alpha = self.alpha_net(alphaControl)
        beta = self.beta_net(betaControl)
        z[:,:,-1] /= dt
        z1 = z[:,:,:1]
        z1_2 = z[:,:,:1]**2 
        z1_3 = z[:,:,:1]**3 
        z2 = z[:,:,1:]

        yhat = alpha + beta.abs()**2 *z1 + z1_2 - z1_3 - z1 * z2/self.tau - z1_2 * z2/self.tau - d 
        return yhat, torch.cat([alphaControl,betaControl,dControl]),torch.zeros((1,),device=self.device)
    
    def get_funcs(self,x,dt):

        B,_,d = x.shape
        # x: x_0, x_dt, x_2dt,...
        smooth_len = int(round(self.smooth_len/dt))

        xdot= deriv_approx_dy(x)
        # dx: dx_4dt,dx_5dt,dx_6dt,..., dx_(l-4)dt
        
        z = torch.cat([x[:,4:-4,:],xdot],dim=-1)
        L = z.shape[1]
        x_in = torch.cat([z, torch.flip(z,[1])],dim=-1)
        alphaControl = self.alphaMamba(x_in)
        betaControl = self.betaMamba(x_in)
        dControl = self.dMamba(x_in)
        
        
        dControl=smooth(dControl.abs(),smooth_len)
        betaControl=smooth(betaControl.abs(),smooth_len)
        #b = nn.ReLU()(sel.b)/model.tau * torch.ones(L,device='cuda')#.view(B,L,d,self.poly_dim,self.poly_dim) # B x L x 2d x P x P
        d = nn.ReLU()(self.d_net(dControl))
        if self.noInput:
            d = torch.zeros(d.shape,device='cuda')
        alpha = self.alpha_net(alphaControl)
        beta = self.beta_net(betaControl)
        alpha *= self.tau**2
        beta *= self.tau
        d *= self.tau**2

        z[:,:,-1] /= dt
        z1 = z[:,:,:1]
        z1_2 = z[:,:,:1]**2 
        z1_3 = z[:,:,:1]**3 
        z2 = z[:,:,1:]
        
        return alpha,beta,d,torch.cat([alphaControl,betaControl,dControl],dim=-1)
    
    def integrate(self,x,dt):

        B,_,d = x.shape
        # x: x_0, x_dt, x_2dt,...
        smooth_len = int(round(self.smooth_len/dt))

        xdot= deriv_approx_dy(x)
        # dx: dx_4dt,dx_5dt,dx_6dt,..., dx_(l-4)dt
        
        z = torch.cat([x[:,4:-4,:],xdot],dim=-1)
        L = z.shape[1]

        alpha,beta,d,states = self.get_funcs(x[:1,:,:],dt)
        alpha,beta,d= alpha.detach().cpu().numpy().squeeze(),beta.detach().cpu().numpy().squeeze(),\
                            d.detach().cpu().numpy().squeeze()
        
        st = 0.05
        start = int(round(st/dt))
        #alpha,beta,d = alpha.detach().cpu().numpy(),beta.detach().cpu().numpy(),d.detach().cpu().numpy()
        alpha,beta,d = alpha[start:],beta[start:],d[start:]

        t_steps = np.arange(0,L*dt+dt/2,dt)[:L][start:]
        alphaTerp = lambda t: np.interp(t,t_steps,alpha)
        betaTerp = lambda t: np.interp(t,t_steps,beta)
        z0 = z[0,start,:]
        z0[-1] /= dt

        def dz(t,z):

            # t: time, should have a timestep of roughly 1. treat as ZOH
            # z: B x 2d
            b_ind = int(t)
            
            alpha_step = alpha[b_ind]#omegaTerp(t)#omega[b_ind]
            beta_step = beta[b_ind] #gammaTerp(t)#gamma[b_ind]
            d_step = d[b_ind]
            #print(b_step.shape)
            #print(t)
            z1 = z[:1]
            #print(z1.shape)
            z1_2 = z[:1]**2  #z[:,:,:1,None].expand(-1,-1,-1,self.poly_dim) # B x L x d -> B x L x 2d x P
            z1_3 = z[:1]**3  #z[:,:,:1,None].expand(-1,-1,-1,self.poly_dim) # B x L x d -> B x L x 2d x P
            z2 = z[1:] #,None].expand(-1,-1,-1,self.poly_dim) # B x L x d -> B x L x 2d x P
    
            #print(power_mat_z1.shape)       
            #power_mat_z1 = power_mat_z1.pow(self.powers)
            #power_mat_z2 = power_mat_z2.pow(self.powers)
            #pow1 = torch.einsum('bldjk,bldj->bldk',b,power_mat_z1)
            dz2 = alpha_step + beta_step.abs()**2 *z1 + z1_2 *self.tau**2 - z1_3*self.tau**2 - z1 * z2*self.tau - z1_2 * z2*self.tau - d_step 
            #dz2 = -(omega_step**2)*z1 - gamma_step * z2 + b_step*power_mat_z1 * z2
            #print(dz2.shape)
            #print(mult * z[0])
            #assert False
            dz1 = z[1]
            
            return np.hstack([dz1,dz2])
        
        #print(t_steps.shape)
        start = time.time()
        obj = solve_ivp(dz,(st,L*dt),z0.squeeze().detach().cpu().numpy(),t_eval=t_steps,method='RK45',atol=1e-5)
        print(f'integrated in {np.round(time.time() - start,2)} s')
        #print(obj.y.shape)
        return obj.y,alpha,beta,d,obj.status