from scipy.integrate import solve_ivp
import numpy as np
import torch
from model.model_utils import smooth
from utils import deriv_approx_d2y,deriv_approx_dy
from data.real_data import get_segmented_audio
import vocalpy as voc
import os

C = 343
L = 0.035
T = L/C 
r = 0.65
Ch = 1.43e-10
MG = 20
MB = 1e4
RB = 5e6
Rh = 24e3

def simulate_trachea_fixed_params(y,dt):

    p_in = []
    p_out = []
    p_in_base = 0
    p_out_base=0
    ly = len(y)
    ts = np.arange(0,ly*dt + dt/2,dt)[:ly]
    for ii,t in enumerate(ts):
        in_shift = int((t - T)/dt)
        out_shift = int((t - T/2)/dt)
        if in_shift < 0:
            p_in_step = p_in_base
        else:
            p_in_step = p_in[in_shift]
        if out_shift < 0:
            p_out_step = p_out_base
        else:
            p_out_step = p_in[out_shift]

        p_in.append(y[ii] - r*p_in_step)
        p_out.append((1 - r)*p_out_step)

    return np.hstack(p_in),np.hstack(p_out)

def simulate_OEC_fixed_params(y,dt):

    dy = np.diff(y)/dt
    y = y[1:]
    i1,i2,i3 = np.zeros(3)
    i0  = np.hstack([i1,i2,i3])
    #i1s,omegas,i3s = [i1],[i2],[i3]
    ly = len(y)
    ts = np.arange(0,ly*dt + dt/2,dt)[:ly]
    def d_oec(t,v):
        ind = min(ly-1,int(t/dt))
        pout = y[ind]
        dpout = dy[ind]
        i1,i2,i3 = v
        di1 = i2
        di2 = -(1/Ch/MG)*i1 - Rh*(1/MB + 1/MG)*i2 + (1/MG/Ch+Rh*RB/MG/MB)*i3 \
                    +(1/MG)*dpout + (Rh*RB/MG/MB)*pout
        di3 = -(MG/MB)*i2 - (Rh/MB)*i3 + (1/MB)*pout

        return np.hstack([di1,di2,di3])

    #s = time.time()
    st=0
    obj = solve_ivp(d_oec,(st,ly*dt),i0,t_eval=ts,method='RK45',atol=1e-5)

    return obj.y[-1,:] * Rh

def integrate(model,x,dt,method='RK45',st=0.05,scaled=True,with_residual=False,use_omega=True,use_gamma=True):

    smooth_len = int(round(model.smooth_len/dt))
    B,_,D = x.shape
    # x: x_0, x_dt, x_2dt,...
    xdot= deriv_approx_dy(x)
    # dx: dx_4dt,dx_5dt,dx_6dt,..., dx_(l-4)dt
    xddot = deriv_approx_d2y(x)

    z = torch.cat([x[:,4:-4,:],xdot],dim=-1)
    L = z.shape[1]

    yhat,*_ = model.forward(x[:1,:,:],dt)
    residual = xddot - yhat 
    smoothed_residual = smooth(residual,smooth_len).detach().cpu().numpy().squeeze()

    omega,gamma,weighted_kernels,states = model.get_funcs(x[:1,:,:],dt,scaled=scaled)
    omega,gamma,weighted_kernels= omega.detach().cpu().numpy().squeeze(),gamma.detach().cpu().numpy().squeeze(),\
                        weighted_kernels.detach().cpu().numpy().squeeze()
    
    start = int(round(st/dt))
    omega,gamma,weighted_kernels,smoothed_residual = omega[start:],gamma[start:],weighted_kernels[start:],smoothed_residual[start:]

    t_steps = np.arange(0,L*dt+dt/2,dt)[:L][start:]
    omegaTerp = lambda t: np.interp(t,t_steps,omega)
    gammaTerp = lambda t: np.interp(t,t_steps,gamma)
    weighted_kernelsTerp = lambda t: np.interp(t,t_steps,weighted_kernels)
    if with_residual:
        residTerp = lambda t: np.interp(t,t_steps,smoothed_residual)
    z0 = z[0,start,:]
    z0[-1] /= dt

    #tau = self.tau#.detach().cpu().numpy()

    def dz(t,z):

        # t: time, should have a timestep of roughly dt. treat as ZOH
        # z: B x 2d
        b_ind = int(t/dt)
        
        if use_omega:
            omega_step = omegaTerp(t) 
        else:
            omega_step = 0
        if use_gamma:
            gamma_step = gammaTerp(t) 
        else:
            gamma_step = 0
        weighted_kernels_step = weighted_kernelsTerp(t) 

        z1 = z[:1]
        
        z2 = z[1:] 

        #-(omega**2)*z1 - gamma * z2 - weighted_kernels
        dz2 = -(omega_step**2)*z1 - gamma_step * z2 - weighted_kernels_step
        dz1 = z[1]
        if with_residual:
            resids = residTerp(t)
            dz2 += resids
        
        return np.hstack([dz1,dz2])
    
    
    obj = solve_ivp(dz,(st,L*dt),z0.squeeze().detach().cpu().numpy(),t_eval=t_steps,method=method,atol=1e-5)
    #print(f'integrated in {np.round(time.time() - s,2)} s')
    if with_residual:
        return obj.y,omega,gamma,weighted_kernels,smoothed_residual,obj.status
    return obj.y,omega,gamma,weighted_kernels,obj.status

def pad_with_nans(x,target_length,axis=0):

    currlen = x.shape[axis]
    currshape = list(x.shape)
    diff = target_length - currlen
    currshape[axis] = diff

    if diff > 0:
        padding = np.full(currshape,np.nan)

        return np.concatenate([x,padding],axis=axis)
    else:
        return x


def get_model_fncs(model,audios,dt,smoothing=True,smooth_len=0.005):

    orig_smooth_len = model.smooth_len
    model.smooth_len = smooth_len

    weights,omegas,gammas,kernels = [],[],[],[]
    for x in audios:

        xdot= torch.from_numpy(deriv_approx_dy(x)).to(model.device).to(torch.float32)

        #print(xdot[:10])
        #print(xdot[:10]/dt)
        # dx: dx_4dt,dx_5dt,dx_6dt,..., dx_(l-4)dt
        xddot = torch.from_numpy(deriv_approx_d2y(x)).to(model.device).to(torch.float32)/dt**2
        sample_ax = np.arange(0,xddot.shape[1]*dt + dt/2,dt)[:xddot.shape[1]]
        #xddot_hat = make_interp_spline(sample_ax,xddot.detach().cpu().numpy().squeeze())

        x = torch.from_numpy(x).to(model.device).to(torch.float32)
     

        #### get functions from model (assumes we only get one x at a time
        omega,gamma,weighted_kernel,weight,states = model.get_funcs(x,xdot,dt,scaled=True,smoothing=smoothing)
        
        omega,gamma,weighted_kernel= omega.detach().cpu().numpy().squeeze(),gamma.detach().cpu().numpy().squeeze(),\
                        weighted_kernel.detach().cpu().numpy().squeeze()
        weight = weight.detach().cpu().numpy().reshape([weight.shape[0],weight.shape[1],-1]).squeeze()
        weights.append(weight)
        kernels.append(weighted_kernel)
        omegas.append(omega)
        gammas.append(gamma)

    model.smooth_len = orig_smooth_len

    return omegas,gammas,kernels,weights

def get_marmo_fncs(model,audio_location,seg_location,marmo_name,voctype='',seed=None):

    audio_location = os.path.join(audio_location,marmo_name,'wavfiles','synchro_cleaned')
    seg_location = os.path.join(seg_location,marmo_name,'wavfiles')

    audios,sr = get_segmented_audio(audio_location,seg_location,audio_subdir='',\
                            seg_subdir='',envelope=False,context_len=0.15,\
                            audio_type=f'{voctype}_cleaned.wav',seg_type=f'{voctype}.txt',\
                                max_pairs=3000,seed=seed,full_vocs=True,extend=False)
    
    audios = [a[0] for a in audios]
    omegas,gammas,kernels,weights = get_model_fncs(model,audios,1/sr)

    return omegas,gammas,kernels,weights

def get_budgie_fncs(model,audio_location,seg_location,seed=None,cut_percentile=75):

    audios,sr = get_segmented_audio(audio_location,seg_location,audio_subdir='',\
                                seg_subdir='',envelope=False,context_len=0.15,\
                                audio_type='_cleaned.wav',seg_type='izationTimestamp.txt',\
                                    max_pairs=3000,seed=seed,full_vocs=True,extend=False)

    audios = audios[0]
    audio_lens = np.array(list(map(lambda y: y.shape[1],audios)))
    audio_lens = np.hstack(audio_lens)

    ### warbles are relatively long -- so take the longest, idk, 25% of vocalizations
    min_len = np.percentile(audio_lens,cut_percentile)
    aud_inds = np.argwhere(audio_lens > min_len)
    audios = [audios[o] for o in aud_inds]

    omegas,gammas,kernels,weights = get_model_fncs(model,audios,1/sr)

    return omegas,gammas,kernels,weights
    

def assess_variability(model,audio_location,seg_location,\
                       audio_filetype='.wav',seg_filetype='.txt',max_samples=5000,\
                        norm_sd=True,return_all=False):


    audios,sr = get_segmented_audio(audio_location,seg_location,\
                        envelope=False,context_len=0.1,max_pairs=max_samples,\
                            audio_type=audio_filetype,seg_type=seg_filetype,full_vocs=True)
    dt = 1/sr

    omegas,gammas = [],[]
    for session in audios:
        for sample in session:
            #print(sample.shape)
            sample = torch.from_numpy(sample).to(model.device).to(torch.float32)
            omega,gamma, *_ = model.get_funcs(sample,dt,scaled=True,smoothing=True)

            omega,gamma =omega.detach().cpu().numpy().squeeze(),gamma.detach().cpu().numpy().squeeze()
            omegas.append(omega)
            gammas.append(gamma)
    max_len = max(list(map(len,omegas)))
    padder = lambda x: pad_with_nans(x,target_length=max_len)

    omegas = np.stack(list(map(padder,omegas)),axis=0)
    gammas = np.stack(list(map(padder,gammas)),axis=0)

    mu_gammas = np.nanmean(gammas,axis=0)
    mu_omegas = np.nanmean(omegas,axis=0)
    sd_gammas = np.nanstd(gammas,axis=0)#/np.abs(np.nanmean(mu_gammas)) # as portion of mean mean!!
    sd_omegas = np.nanstd(omegas,axis=0)#/np.abs(np.nanmean(mu_omegas)) # as portion of mean mean!!

    if norm_sd:

        sd_gammas/= np.abs(np.nanmean(mu_gammas))
        sd_omegas /= np.abs(np.nanmean(mu_omegas))

    if return_all:
        return (mu_omegas,mu_gammas), (sd_omegas,sd_gammas),(omegas,gammas)
    else:
        return (mu_omegas,mu_gammas), (sd_omegas,sd_gammas)
    
def feature_variability(audio_location,seg_location,\
                       audio_filetype='.wav',seg_filetype='.txt',max_samples=5000,\
                        norm_sd=True,return_all=False):

    audios,sr = get_segmented_audio(audio_location,seg_location,\
                        envelope=False,context_len=0.1,max_pairs=max_samples,\
                            audio_type=audio_filetype,seg_type=seg_filetype,full_vocs=True)

    
    features = {}
    for session in audios:
        for audio in session:
            #print(len(audio))
            a = voc.Sound(audio.squeeze(),samplerate=sr)
            feats = voc.feature.sat.similarity_features(a)

            for vn,da in feats.data.items():
                try:
                    features[vn].append(da.to_numpy().squeeze())
                except:
                    features[vn] = [da.to_numpy().squeeze()]
                    t_len = len(audio.squeeze())/sr
                    #print(len,int,round)
                    sr_new = int(round(len(features[vn])/t_len))
                    #print(t_len,sr_new)
        
    for feat in features.keys():
        features[feat] = np.stack(features[feat],axis=0)
    return features,features.keys(),sr_new

def calc_vi(data):

    # expects data to be nsamples x time x nfeatures
    median = np.nanmedian(data,axis=0,keepdims=True)
    VI = np.amin(np.linalg.norm(data - median,axis=-1)**2,axis=0)

    return VI

