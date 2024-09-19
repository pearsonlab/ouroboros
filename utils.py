import torch
import numpy as np

from scipy.signal.windows import hann
from scipy.signal import ShortTimeFFT as STFFT
from scipy.interpolate import RegularGridInterpolator

def from_numpy(data,device='cuda'):

    return torch.from_numpy(data).type(torch.FloatTensor).to(device)

def remove_axes(axis):

    axis.set_xticks([])
    axis.set_yticks([])

def get_spec(audio,fs,onset,offset,shoulder=0.05,n_freq_bins = 64):

    """
    make a spectrogram for a given vocalization.
    """
    flag = 0
    if offset - onset <= 0.05:
        flag = 1
    audiotimes = np.linspace(0,len(audio)/fs,len(audio))
    a,t = audio[np.searchsorted(audiotimes,onset-shoulder):np.searchsorted(audiotimes,offset+shoulder)],\
        audiotimes[np.searchsorted(audiotimes,onset-shoulder):np.searchsorted(audiotimes,offset+shoulder)]
    N = len(t)
    w = hann(128, sym=True)  # symmetric Gaussian window

    transform = STFFT(w,hop=16,fs=fs,mfft = 1028)

    Sx = transform.stft(a)
    
    t_lo,t_hi,f_lo,f_hi = transform.extent(N)
    tAx = np.linspace(t_lo,t_hi,Sx.shape[1]) + onset
    
    t0,t1 = np.searchsorted(tAx,onset),np.searchsorted(tAx,offset)

    Sx,tAx = Sx[:,t0:t1],tAx[t0:t1]

    fAx = np.linspace(f_lo,f_hi,Sx.shape[0])
    target_freqs = np.linspace(f_lo,f_hi,n_freq_bins)
    target_ts = np.arange(tAx[0],tAx[-1],0.001)

    Sx = np.log(np.abs(Sx) + 1e-12)
    Sx = (Sx + 6.5) / 5
    Sx = np.clip(Sx, 0.0, 1.0)

    interp = RegularGridInterpolator((fAx,tAx),Sx,bounds_error=True,fill_value=-1e10)

    newX,newY = np.meshgrid(target_freqs,target_ts,indexing='ij',sparse=True)

    Sx2 = interp((newX,newY))
    Sx = Sx2

    
    return Sx,target_ts,target_freqs,flag