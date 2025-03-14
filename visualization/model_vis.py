
import numpy as np
from utils import get_spec,deriv_approx_d2y,deriv_approx_dy,from_numpy
import torch 
import matplotlib.pyplot as plt
import gc
from model import model_utils
from matplotlib.colors import Normalize
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from scipy.io import wavfile
import librosa
from itertools import repeat
from matplotlib.colors import ListedColormap
import os
plt.rcParams['text.usetex'] = True


def format_axes(ax,xlabel='',ylabel='',xlims=(),ylims=()):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(axis='both',which='both',direction='in')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(xlims) == 2:
        ax.set_xlim(xlims)
    if len(ylims) == 2:
        ax.set_ylim(ylims)
    return 

def loss_plot(train_loss,val_loss,save_loc='',show=True):

    train_loss,val_loss = np.array(train_loss),np.array(val_loss)

    ax = plt.gca()
    ax.plot(train_loss,color='tab:blue',label="Train loss")
    ax.plot(val_loss[:,0],val_loss[:,1],color='tab:orange',label ='Validation loss')
    #ax.set_xlabel("Gradient steps")
    #ax.set_ylabel("Loss (MSE)")

    format_axes(ax,xlabel="Gradient steps",ylabel="Model performance (R2)",ylims=(0,1))
    plt.legend()
    plt.savefig(os.path.join(save_loc,'train_test_loss.svg'))
    if show:
        plt.show()
    
    plt.close()

def visualize_kernel(model,y,dt):

    # y: B x L x d
    y = y[None,:,:]
    smooth_len = int(round(model.smooth_len/dt))
    dy = deriv_approx_dy(y)
    z = torch.cat([y[:,4:-4,:],dy],dim=-1)
    x_in = torch.cat([z, torch.flip(z,[1])],dim=-1)

    weight_control = model.kernel_mamba(x_in)
    weighted_kernels = model.kernel(z,weight_control,smooth_len)/model.tau
    y,dy = y.detach().cpu().numpy().squeeze()[4:-4],dy.detach().cpu().numpy().squeeze()/dt
    spec,t,f, _ = get_spec(y,int(round(1/dt)),onset=0,offset=len(y)*dt,shoulder=0.0,interp=False,win_len=1028,min=-2,max=3.5)
    weighted_kernels = weighted_kernels.detach().cpu().numpy().squeeze()

    fig,axs = plt.subplots(nrows=2,ncols=1,figsize=(6,10))
    g = axs[1].scatter(y,dy,c=weighted_kernels)
    axs[1].set_xlabel("y")
    axs[1].set_ylabel("dy")
    cb = plt.colorbar(g,ax=axs[1])
    cb.set_label("Weighted kernel value",rotation=270,labelpad=20)
    axs[0].imshow(spec,vmin=0,vmax=1,origin='lower',extent=[t[0],t[-1],f[0],f[-1]],aspect='auto')
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

#### gif plot of y vs ydot, omega vs gamma, avged kernel


def grab_segments(seg_list,*args,fs=42000):

    #### IMPLEMENT
    output = [[] for _ in args]
    for ii, onoffs in enumerate(seg_list):
        #onoffs = np.loadtxt(seg_file)
        if np.ndim(onoffs) == 1:
            onoffs = onoffs[None,:]
        for onInd,offInd in onoffs:
            #onInd, offInd = int(round(on*fs)),int(round(off*fs))
            for jj,inputs in enumerate(args):
                #print(inputs[ii].squeeze()[onInd:offInd].shape)
                output[jj].append(inputs[ii][onInd:offInd])

        #print(len(output[0]))
        #assert False
        #print(output)
    return output

def make_gif(audio_list,model,n_sylls,dt,grid_size=100,max_len=1000,batch_size=64,save_name='./test_gif.gif',step=1,plot_length=100,audio_type='full files',seg_list=[]):
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    omega_length = plot_length // 2
    #### get real data (y, dy)
    if audio_type == 'full files':
        ys = audio_list
        min_len = min(list(map(len,ys)))
        ys  = [y[:min_len] for y in ys]
        dys = [deriv_approx_dy(a[None,:,None]).squeeze()[:min_len] for a in ys]
    elif audio_type == 'segmented':
        # this is just going to assume we've already selected out the audio files that are segmented....easier for now
        ys = []
        segs = []
        assert len(seg_list) > 0, print("if using segmented audio, need some segment files!")
        for a in audio_list:

            fs, a = wavfile.read(a)
            if a.dtype == np.int16:
                a = a/-np.iinfo(a.dtype).min
            
            ys.append(a)
        dys = [deriv_approx_dy(a[None,:,None]).squeeze() for a in ys]
        min_len=len(a)
        for seg in seg_list:
            onoffs = np.round(np.loadtxt(seg)*fs).astype(np.int32) - 4
            onoffs[:,0] -= 1000
            onoffs[:,1] += 1000
            if np.ndim(onoffs) == 1:
                onoffs = onoffs[None,:]
            segs.append(onoffs)
            lens = onoffs[:,1] - onoffs[:,0]
            if np.amin(lens) < min_len:
                min_len = np.amin(lens)
        #print(segs)
        #print(min_length)
        #assert False
    else:
        raise NotImplementedError

    
    
    #min_len = min(list(map(len,ys)))
    #ys  = [y[:min_len] for y in ys]
    t = np.arange(0,min_len*dt + dt/2,dt)[:min_len]
    maxy = max(list(map(lambda x: np.amax(np.abs(x)),ys)))
   
    
    smooth_len = int(round(model.smooth_len/dt))
    fs = int(round(1/dt))
    
    maxdy = max(list(map(lambda x: np.amax(np.abs(x)),dys)))

    if max_len == None:
        max_len = min_len
    max_len = max_len // step
    ##### get learned functions from model
    print('getting model functions...')
    funs = [model.get_funcs(from_numpy(a[None,:,None]),dt,scaled=True) for a in ys]

    omegas,gammas,kernels,controls = [f[0] for f in funs],[f[1] for f in funs],\
                                    [f[2] for f in funs], [f[3] for f in funs]

    
        #assert False
    #print(list(map(len,omegas)))
    #print(list(map(np.shape,omegas)))
    #print(min_len)
    #assert False
    omegas,gammas,kernels= [model_utils.smooth(omega,smooth_len).detach().cpu().numpy().squeeze() for omega in omegas],\
                            [model_utils.smooth(gamma,smooth_len).detach().cpu().numpy().squeeze() for gamma in gammas],\
                            [model_utils.smooth(kernel,smooth_len).detach().cpu().numpy().squeeze() for kernel in kernels]
    if audio_type == 'segmented':
        [omegas,gammas,kernels] = grab_segments(segs,omegas,gammas,kernels,fs = int(round(1/dt)))

    omegas,gammas,kernels = [omega[:min_len][::step] for omega in omegas],\
                            [gamma[:min_len][::step] for gamma in gammas],\
                            [kernel[:min_len][::step] for kernel in kernels]
    #print(len(omegas))
    #assert False
    kernel_max = max(list(map(lambda x: np.amax(np.abs(x)),kernels)))
    gamma_max = max(list(map(lambda x: np.amax(np.abs(x)),gammas)))
    omega_max = max(list(map(lambda x: np.amax(np.abs(x)),omegas)))
    print('done!')
    #print(kernel_max)
    
    #print([omega.shape for omega in omegas])
    #print([gamma.shape for gamma in gammas])
    #assert False
    #print([k]
    ####### Set up control functions for model
    ys = [y[4:-4] for y in ys]
    B,L,d = ys[0][None,:,None].shape
    #min_len -= 8 
    #print(list(map(np.shape,dys)),list(map(np.shape,ys)))
    zs = [torch.cat([from_numpy(y)[None,:,None],from_numpy(dy)[None,:,None]],dim=-1) for y,dy in zip(ys,dys)] 
    x_ins = [torch.cat([z, torch.flip(z,[1])],dim=-1) for z in zs]

    weight_controls = [model.kernel_mamba(x_in) for x_in in x_ins]
    ####### get average kernel function for all time points
    weights = [model.kernel.get_weights(weight_control,smooth_len).detach().cpu().numpy().squeeze() for weight_control in weight_controls]
    
    if audio_type == 'segmented':
        #print(list(map(np.shape,ys)))
        #print(list(map(np.shape,dys)))
        #print(list(map(np.shape,weights)))
        [ys,dys,weights] = grab_segments(segs,ys,dys,weights,fs=int(round(1/dt)))
        ys,dys = [y.squeeze()[:min_len] for y in ys],[dy.squeeze()[:min_len] for dy in dys]
        weights = [w.squeeze()[:min_len] for w in weights]
    #print(list(map(len,weights)))
    #print(len(weights))
    #assert False
    ygrid,dygrid = np.linspace(-maxy,maxy,grid_size), np.linspace(-maxdy,maxdy,grid_size)
    
    ygrid,dygrid = np.meshgrid(ygrid,dygrid)
    ygrid,dygrid = ygrid.flatten(),dygrid.flatten()
    zgrid = np.stack([ygrid,dygrid],axis=-1)

    kernel_grids = []
    print('getting kernel grids....')

    trajs_2d = [np.stack([o,g],axis=1) for o,g in zip(omegas,gammas)]
    diffs_2d = [np.diff(traj,axis=0) for traj in trajs_2d]
    trajs_4d = [np.hstack([t[1:],d]) for t, d in zip(trajs_2d,diffs_2d)]

    base_index = np.random.choice(len(trajs_4d))
    base_traj = trajs_4d[base_index]
    distances = [librosa.sequence.dtw(X=t1.T,Y=base_traj.T,metric='cosine')[0][-1,-1] for t1 in trajs_4d]

    color_map_distance = Normalize(0,np.amax(distances))
    cm1 = cm.plasma
    omega_gamma_colors = [cm1(color_map_distance(d)) for d in distances]
    omega_gamma_map = cm.ScalarMappable(norm=color_map_distance,cmap=cm1)
    #print([cm1(color_map_distance(d)) for d in distances])
    #assert False
    
    for ws in weights:
        #print(ws.shape)
        #print(zgrid.shape)
        L = len(ws)
        kernel_grid=[]
        for on in range(0,L,batch_size):
            off = min(L,on+batch_size)
            tmpL = off - on
            zgridTiled = np.tile(zgrid[None,:,:],(tmpL,1,1)) # switching Length and Batch size dimensions, so we get full grid at each time point
            wsTiled = np.tile(ws[on:off,None,None,:],(1,len(zgrid),1,1))
            try:
                grid_slice = model.kernel.forward_given_weights(from_numpy(zgridTiled),from_numpy(wsTiled).to(torch.float64)).detach().cpu().numpy().squeeze()
                ## for bug fixing
                #grid_slice = forward_given_weights(model.kernel,from_numpy(zgridTiled),from_numpy(wsTiled).to(torch.float64)).detach().cpu().numpy().squeeze()
            except:
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()
                assert False
            kernel_grid.append(grid_slice)
        kernel_grid = np.concatenate(kernel_grid,axis=0)
        kernel_grids.append(kernel_grid)

    #print(list(map(np.shape,kernel_grids)))
    #assert False
    
    ygrid=ygrid.reshape(grid_size,grid_size)
    dygrid = dygrid.reshape(grid_size,grid_size)
    kernel_grids = [kernel_chunk.reshape(min_len,grid_size,grid_size) for kernel_chunk in kernel_grids]

    print('done!')
    ####### Finally, set up plotting for gif ######

    kernel_grids = np.nanmean(np.stack(kernel_grids,axis=0),axis=0)[::step]

    kernel_max = np.amax(np.abs(kernel_grids))
    norm = Normalize(-kernel_max,kernel_max)
    
    specs = [get_spec(y.squeeze(),fs,onset=t[0],offset=t[-1],shoulder=0.0,interp=False,win_len=1028,normalize=False,min=-2,max=3.5,spec_type='log') for y in ys] # gets spec, ts, fs
    ts,fs = [spec[1] for spec in specs], [spec[2] for spec in specs]
    specs = [spec[0] for spec in specs]
    meanSpec = np.nanmean(np.stack(specs,axis=0),axis=0)
    #print(t.shape,meanSpec.shape,omegas[0].shape)
    #ax = plt.gca()
    #ax.imshow()
    ### need to fix blank spec for this fig
    #assert False
    t = t[::step]
    print(t[0],ts[0][0])
    print(t[-1],ts[0][-1])
    #print(list(map(np.shape,omegas)))
    #print(list(map(np.shape,gammas)))
    fig,axs = plt.subplots(nrows=1,ncols=4,figsize=(18,5))
    def init(axs):

        ### change each of the line plots to a scatter, with alpha by how long ago it was plotted. maybe additionally: calculate d omega, d gamma, color by distance in 4d space?
        axs[0].imshow(meanSpec,origin='lower',aspect='auto',extent=[ts[0][0],ts[0][-1],fs[0][0],fs[0][-1]],vmin=-5,vmax=4,zorder=-1)
        xlims = axs[0].get_xlim()
        ylims = axs[0].get_ylim()
        vline = axs[0].axvline(t[0],color='tab:red',zorder=4)
        format_axes(axs[0],ylabel='Frequency (Hz)',xlabel='Time (s)',xlims=xlims)

        ydyLines = []
        ydycmaps = []
        for ii,(y,dy) in enumerate(zip(ys,dys)):

            color0 = cm.tab10(ii)
            alpha_fnc = lambda x: min(1,np.exp(-x/50))
            ydynorm = Normalize(-plot_length,0)
            color_list = [(c[0],c[1],c[2],alpha_fnc(ii)) for c,ii in zip(repeat(color0),range(201))]
            new_cmap = ListedColormap(color_list)
            
            #l = axs[1].scatter(y[0],dy[0]/dt,alpha=1,c=[0],cmap=new_cmap,norm=ydynorm)
            l, = axs[1].plot(y[0],dy[0]/dt)
            ydyLines.append(l)
            #ydycmaps.append(new_cmap)
            #axs[0].scatter(y[0],dy[0]/dt,s=10)
        format_axes(axs[1],ylabel='dy',xlabel='y',xlims=(-maxy,maxy),ylims=(-maxdy/dt,maxdy/dt))
            
        ogLines=[]
        ogcmaps = [] 
        for omega,gamma,cols in zip(omegas,gammas,omega_gamma_colors):

            alpha_fnc = lambda x: min(1,np.exp(-x/100))
            ognorm = Normalize(-omega_length,0)
            color_list = [(c[0],c[1],c[2],alpha_fnc(ii)) for c,ii in zip(repeat(cols),range(201))]
            new_cmap = ListedColormap(color_list)
            
            l = axs[2].scatter(omega[0],gamma[0],norm=ognorm,cmap=new_cmap,c=[0])
            ogLines.append(l)
            #axs[1].scatter(omega[0],gamma[0],s=10)
        format_axes(axs[2],xlabel='omega',ylabel='gamma',xlims=(0,omega_max),ylims=(-gamma_max,gamma_max))
        cb_og = plt.colorbar(omega_gamma_map,ax=axs[2],label=f'DTW distance from trajectory {base_index + 1}')
        
        surf = axs[3].pcolormesh(ygrid,dygrid/dt,kernel_grids[0],cmap=cm.coolwarm,norm=norm)
        format_axes(axs[3],xlabel='y',ylabel='dy')
        plt.colorbar(surf,ax=axs[3])
        
        return vline,ydyLines,ogLines,surf,ydycmaps,ogcmaps

    #### pause for 2s at end
    ## i is the frame number
    vl,ydyl,ogl,hm,ycm,ogcm = init(axs)
    fig.tight_layout()
    #print(vl.get_xdata())
    #assert False
    #plt.show()
    #plt.close()
    ys = [y[::step] for y in ys]
    dys = [dy[::step] for dy in dys]

    ### something here is fucky -- points aren't plotting anymore for some reason......
    ### was the scale messed up? is alpha just 0 for everything?
    ### both ydy and omega gamma plots are off
      
    def animate(i): 
        #fig,axs = plt.subplots(nrows=1,ncols=3,figsize=(15,5))
        #print(i,end='-')
        i #*= step
        plot_end = max(1,min(i,min_len))
        print(plot_end,end='-')
        plot_start = max(0,plot_end - plot_length)
        omega_start = max(0,plot_end - omega_length)
        #if i >= 60:
        #    print(np.arange(0,plot_start - plot_end - 1,-1))
        #    print(
        ## update lines using set_data. colors should remain the same as when initialized. 
        vl.set_xdata([t[plot_end],t[plot_end]])
        for l,y,dy in zip(ydyl,ys,dys):
            
            #### Line plot #############
            l.set_data(y[plot_start:plot_end],dy[plot_start:plot_end]/dt) 

            ######## Scatter plot ###########
            #l.set_offsets(np.stack([y[:plot_end],dy[:plot_end]/dt],axis=1))
            #l.set_array(np.arange(0,plot_start - plot_end - 1,-1))
        for l,omega,gamma in zip(ogl,omegas,gammas):
            #l.set_data(omega[omega_start:plot_end],gamma[omega_start:plot_end])
            l.set_offsets(np.stack([omega[omega_start:plot_end],gamma[omega_start:plot_end]],axis=1))
            l.set_array(np.arange(0,omega_start- plot_end-1,-11))
        
        ## eventually, should allow for list of colors

        ## for the heatmap, update data with .set_array(grid.ravel()) (could probably use flatten also)
        hm.set_array(kernel_grids[plot_end-1].ravel())
        #plt.show()
        return ydyl + ogl + [hm]
                
    #print(kernel_grids.shape)
    
    #print(ygrid,dygrid)
    #interact(animate,i=(0,min_len,1))
    #print([kernel_grid.shape for kernel_grid in kernel_grids])
    min_len = min_len // step
    print(min_len)
    assert min_len > 1
    #assert False
    print('animating....')
    ani = FuncAnimation(fig,func=animate,frames=max_len + 400,interval=10)
    ani.save(save_name,writer='ffmpeg')
    print('done!')
    plt.close()
             
    return





        