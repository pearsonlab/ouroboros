from utils import deriv_approx_dy,from_numpy
import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(model,x,dt):

    xdot= deriv_approx_dy(x)
    # dx: dx_4dt,dx_5dt,dx_6dt,..., dx_(l-4)dt

    z = torch.cat([x[:,4:-4,:],xdot],dim=-1)
    L = z.shape[1]
    x_in = torch.cat([z, torch.flip(z,[1])],dim=-1)
    states = model.controlMamba(model.control_proj(x_in))
    states = torch.flip(states,[1])
    
    #nn.ReLU()(self.b_net(states))#.view(B,L,d,self.poly_dim,self.poly_dim) # B x L x 2d x P x P
    d = nn.ReLU()(model.d_net(states))
    if model.noInput:
        d = torch.zeros(d.shape,device='cuda')
    omega = model.omega_net(states)
    gamma = model.gamma_net(states)
    
    omega *= model.tau
    gamma *= model.tau
    #gamma *= 0
    d *= model.tau**2
    b = nn.ReLU()(model.b_net(states)) #model.b_net(states)#.view(B,L,d,self.poly_dim,self.poly_dim) # B x L x 2d x P x P
    b *= model.tau
    
    b = b.detach().cpu().numpy().squeeze()#*model.tau**2
    omega = omega.detach().cpu().numpy().squeeze()#*model.tau
    gamma = gamma.detach().cpu().numpy().squeeze()#*model.tau**2
    d = d.detach().cpu().numpy().squeeze()#*model.tau**2
    #pows = model.powers.detach().cpu().numpy()
    t_steps = np.arange(0,L*dt + dt/2,dt)[:L]
    omegaTerp = lambda t: np.interp(t,t_steps,omega)
    gammaTerp = lambda t: np.interp(t,t_steps,gamma)
    z0 = z[0,0,:]
    def dz(t,z):

        # t: time, should have a timestep of roughly 1. treat as ZOH
        # z: B x 2d
        b_ind = int(t)
        b_step = b[b_ind]
        omega_step = omega[b_ind]#omegaTerp(t)#omega[b_ind]
        gamma_step = gamma[b_ind] #gammaTerp(t)#gamma[b_ind]
        #print(b_step.shape)
        #print(t)
        z1 = z[:1]
        #print(z1.shape)
        power_mat_z1 = z[:1]**2 #z[:,:,:1,None].expand(-1,-1,-1,self.poly_dim) # B x L x d -> B x L x 2d x P
        z2 = z[1:]#,None].expand(-1,-1,-1,self.poly_dim) # B x L x d -> B x L x 2d x P
 
        #print(power_mat_z1.shape)       
        #power_mat_z1 = power_mat_z1.pow(self.powers)
        #power_mat_z2 = power_mat_z2.pow(self.powers)
        #pow1 = torch.einsum('bldjk,bldj->bldk',b,power_mat_z1)
        dz2 = -(omega_step**2)*z1 - gamma_step * z2 + b_step*power_mat_z1 * z2
        #print(dz2.shape)
        #print(mult * z[0])
        #assert False
        dz1 = z[1]
        
        return np.hstack([dz1,dz2])
    #print(L)
    print(t_steps.shape)
    start = time.time()
    obj = solve_ivp(dz,(0,L*dt),z0.squeeze().detach().cpu().numpy(),t_eval=t_steps,method='RK45',atol=1e-5)
    print(f'integrated in {np.round(time.time() - start,2)} s')
    #print(obj.y.shape)
    return obj.y,omega,gamma,b,d,obj.status