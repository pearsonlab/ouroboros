
import numpy as np
from utils import get_spec,deriv_approx_d2y,deriv_approx_dy,from_numpy
import torch 
import matplotlib.pyplot as plt


plt.rcParams['text.usetex'] = True
def get_funcs(model,x,dt):

    B,_,d = x.shape
    # x: x_0, x_dt, x_2dt,...

    xdot= deriv_approx_dy(x)/dt
    # dx: dx_4dt,dx_5dt,dx_6dt,..., dx_(l-4)dt

    z = torch.cat([x[:,4:-4,:],xdot],dim=-1)
    L = z.shape[1]
    x_in = torch.cat([z, torch.flip(z,[1])],dim=-1)
    states = model.controlMamba(model.control_proj(x_in))
    states = torch.flip(states,[1])
    
    b = model.b_net(states).view(B,L,d,model.poly_dim,model.poly_dim) # B x L x 2d x P x P
    omega = model.omega_net(states)
    gamma = model.gamma_net(states)
        
    b *= model.tau**2
    omega *= model.tau**2
    gamma *= model.tau**2

    z1 = z[:,:,:1]
    z2 = z[:,:,1:]
    power_mat_z1 = z[:,:,:1].pow(2) #expand(-1,-1,-1,model.poly_dim) # B x 2d -> B x 2d x P

    b_out =b * power_mat_z1 * z2

    
    return omega,gamma,b,b_out,states


def format_axes(ax,xticks=[],yticks=[]):

    ax.spines[['right','top']].set_visible(False)
    ax.tick_params(axis='both',which='both',direction='in')
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

def generate_quantities_figure(model,audioData,sr,nSamples=10):

    samples = np.random.choice(len(audioData),nSamples)#,p=weights)
    dt = 1/sr
    for jj,s in enumerate(samples):

        aud = audioData[s]
        spec,time,freq,_ = get_spec(aud[:,-1],sr,onset=4*dt,offset=(len(aud) - 4)/sr,shoulder=0.0,interp=False)

        aud = aud[None,:,:]
        dy = deriv_approx_dy(aud)
        d2y = deriv_approx_d2y(aud)
        aud = aud[:,4:-4,:]
        L = aud.shape[1]

        omega,gamma,b,b_out,states = get_funcs(model,from_numpy(aud),dt)
        omega,gamma,b,b_out,states = omega.detach().cpu().numpy(),gamma.detach().cpu().numpy(),\
                                    b.detach().cpu().numpy(),b_out.detach().cpu().numpy(),states.detach().cpu().numpy()
        
        t = np.arange(4*dt,len(aud)*dt + 4*dt,dt)[:L]
        fig,axs = plt.subplots(2,1,figsize=(20,20))

        onInd = int(round(0.1*sr))
        offInd = int(round(0.2*sr))
        onIndSpec,offIndSpec = np.searchsorted(time,0.1),np.searchsorted(time,0.2)

        axs[1].plot(t[onInd:offInd],omega.squeeze()[onInd:offInd],label=r'$\omega$')
        axs[1].plot(t[onInd:offInd],gamma.squeeze()[onInd:offInd],label=r'$\gamma$')
        axs[1].plot(t[onInd:offInd],b.squeeze()[onInd:offInd],label=r'nonlinear correction')

        axs[1].legend(frameon=False)
        format_axes(axs[1],xticks=t[onInd:offInd:200])

        axs[0].imshow(spec[:,onIndSpec:offIndSpec],extent=[0.1,0.2,freq[0],freq[-1]],origin='lower',vmin=0,vmax=1,aspect='auto')
        format_axes(axs[0])






        