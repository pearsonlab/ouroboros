
import numpy as np
from utils import get_spec,deriv_approx_d2y,deriv_approx_dy,from_numpy
import torch 
import matplotlib.pyplot as plt


plt.rcParams['text.usetex'] = True

def format_axes(ax,xticks=[],yticks=[]):

    ax.spines[['right','top']].set_visible(False)
    ax.tick_params(axis='both',which='both',direction='in')
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

def loss_plot(train_loss,val_loss):

    train_loss,val_loss = np.array(train_loss),np.array(val_loss)

    ax = plt.gca()
    ax.plot(train_loss[:,0],color='tab:blue',label="Train loss")
    ax.plot(val_loss[:,0],val_loss[:,1],color='tab:orange',label ='Validation loss')
    ax.set_xlabel("Gradient steps")
    ax.set_ylabel("Loss (MSE)")

    format_axes(ax,xticks=ax.get_xticks(),yticks=ax.get_yticks())
    plt.show()
    plt.close()

def quantities_figure(model,audioData,sr,nSamples=10,save=False,filename='./quants.svg'):

    samples = np.random.choice(len(audioData),nSamples)
    dt = 1/sr
    for jj,s in enumerate(samples):

        aud = audioData[s]
        spec,spec_t,freq,_ = get_spec(aud[:,-1],sr,onset=4*dt,offset=(len(aud) - 4)/sr,shoulder=0.0,interp=False)

        aud = aud[None,:,:]
        dy = deriv_approx_dy(aud)
        d2y = deriv_approx_d2y(aud)
        aud = aud[:,4:-4,:]
        L = aud.shape[1]

        terms = model.get_funcs(from_numpy(aud),dt)
        terms = [t.detach().cpu().numpy() for t in terms]
        omega,gamma,b,b_out,states = model.get_funcs(from_numpy(aud),dt)
        omega,gamma,b,b_out,states = omega.detach().cpu().numpy(),gamma.detach().cpu().numpy(),\
                                    b.detach().cpu().numpy(),b_out.detach().cpu().numpy(),states.detach().cpu().numpy()
        
        t = np.arange(4*dt,len(aud)*dt + 4*dt,dt)[:L]
        fig,axs = plt.subplots(2,1,figsize=(20,20))

        on,off = 0.1,0.2
        onInd = int(round(on*sr))
        offInd = int(round(off*sr))
        onIndSpec,offIndSpec = np.searchsorted(spec_t,on),np.searchsorted(spec_t,off)

        fig,axs = plt.subplotrs(nrows=len(model.names),ncols=1,figsize=(10,3*len(model.names)))
        for ax,term,name in zip(axs[1:],terms,model.names):
            ax.plot(t[onInd:offInd],term.squeeze()[onInd:offInd],label=name)
            ax.legend(frameon=False)
            format_axes(ax,xticks=ax.get_xticks(), yticks = ax.get_yticks())

        axs[0].imshow(spec[:,onIndSpec:offIndSpec],extent=[on,off,freq[0],freq[-1]],origin='lower',vmin=0,vmax=1,aspect='auto')
        format_axes(axs[0])

    plt.savefig(filename)
    plt.show()
    plt.close()







        