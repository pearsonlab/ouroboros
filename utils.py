import torch
import numpy as np

from scipy.signal.windows import hann
from scipy.signal import ShortTimeFFT as STFFT
from scipy.interpolate import RegularGridInterpolator

def deriv_approx_dy(y):

    B,L,d = y.shape
    assert L >= 5, print(f"y is not long enough to approximate with 9 points! needs 9 points, has {L}")
    return (3*y[:,:-8,:]-32*y[:,1:-7,:] + 168 * y[:,2:-6,:] - 672*y[:,3:-5,:] +\
             672*y[:,5:-3,:] - 168*y[:,6:-2,:] + 32*y[:,7:-1,:] - 3*y[:,8:,:])/840

def deriv_approx_d2y(y):

    B,L,d = y.shape
    assert L >= 5, print(f"y is not long enough to approximate with 9 points! needs 9 points, has {L}")
    return (-9 * y[:,:-8,:] + 128*y[:,1:-7,:] -1008*y[:,2:-6,:] + 8064*y[:,3:-5,:]- 14350*y[:,4:-4,:] + \
            8064*y[:,5:-3,:] - 1008*y[:,6:-2,:] + 128* y[:,7:-1,:] - 9*y[:,8:,:])/5040

def from_numpy(data,device='cuda'):

    return torch.from_numpy(data).type(torch.FloatTensor).to(device)

def remove_axes(axis):

    axis.set_xticks([])
    axis.set_yticks([])

def get_spec(audio,fs,onset,offset,shoulder=0.05,n_freq_bins = 64,win_len=128,interp=True,min=-6.5,max=5):

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
    w = hann(win_len, sym=True)  # symmetric Gaussian window

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
    Sx = (Sx - min) / max
    Sx = np.clip(Sx, 0.0, 1.0)

    if interp:
        interp = RegularGridInterpolator((fAx,tAx),Sx,bounds_error=True,fill_value=-1e10)

        newX,newY = np.meshgrid(target_freqs,target_ts,indexing='ij',sparse=True)

        Sx2 = interp((newX,newY))
        Sx = Sx2
    else:
        target_ts,target_freqs = tAx,fAx

    
    return Sx,target_ts,target_freqs,flag


from scipy import signal

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5,axis=-1):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data,axis=axis)
    return y