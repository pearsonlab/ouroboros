import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.special import gamma
import glob
import os
from scipy.interpolate import make_interp_spline

def gen_pure_tones(n_samples=8000,sample_rate=44100,frequency=5000,sample_length=0.3,random_phase=True):


    samples = []

    gen = np.random.default_rng()
    dt = 1/sample_rate
    length = int(round(sample_length * sample_rate))

    for ii in range(n_samples):

        if random_phase:
            start_time = gen.uniform(low=0,high=1)
        else:
            start_time=0

        t = np.arange(start_time,start_time + sample_length + dt/2,dt)

        y = np.sin(2*np.pi * frequency * t)

        samples.append(y[:length,None])

    return np.stack(samples,axis=0)

def gen_mixed_tones(n_samples=8000, sample_rate=44100,min_freq=1000,max_freq=20000,sample_length=0.3,random_phase=True):

    samples = []

    gen = np.random.default_rng()
    dt = 1/sample_rate
    length = int(round(sample_length * sample_rate))
    #max_freq = sample_rate

    for ii in range(n_samples):

        if random_phase:
            start_time = gen.uniform(low=0,high=1)
        else:
            start_time=0

        frequency=np.random.choice(max_freq-1000) + 1000

        t = np.arange(start_time,start_time + sample_length + dt/2,dt)

        y = np.sin(2*np.pi * frequency * t)

        samples.append(y[:length,None])

    return np.stack(samples,axis=0)

def gen_fm_data(n_samples,sample_rate=44100,frequency=5000,sample_length=0.3,sweep_rate=1000):



    samples = []

    gen = np.random.default_rng()
    dt = 1/sample_rate
    length = int(round(sample_length * sample_rate))
    sweep_length = sweep_rate * sample_length

    for ii in range(n_samples):

        start_time = gen.uniform(low=0,high=1)
        sweep_dir = gen.choice([-1,1],1)
        freqs = np.linspace(frequency,frequency + sweep_length*sweep_dir,length)

        t = np.arange(start_time,start_time + sample_length + dt,dt)

        y = np.sin(2*np.pi * freqs * t)

        samples.append(y[:length,None])

    return np.stack(samples,axis=0)

def make_harmonic_stack(fundamental_freq,n_harmonics,t):

    A = np.sin(np.linspace(0,np.pi,len(t)))
    #fig,axs = plt.subplots(nrows=1,ncols=2,figsize=(10,5))
    #axs[0].plot(t,A)
    #axs[1].plot(t,fundamental_freq)
    #axs[0].set_xlabel("Time (s)")
    #axs[1].set_xlabel("time (s)")
    #axs[0].set_ylabel("Amplitude")
    #axs[1].set_ylabel("Fundamental frequency")
    #plt.tight_layout()
    #plt.show()
    #plt.close()

    signal = np.zeros(t.shape)
    d2_signal = np.zeros(t.shape)
    d_signal = np.zeros(t.shape)
    for harm in range(1,n_harmonics+1):

        signal += np.sin(2*np.pi * t * fundamental_freq *harm) * A/harm
        d2_signal += - (fundamental_freq *harm)**2 * np.sin(2*np.pi * t * fundamental_freq *harm) * A/harm
        d_signal += (fundamental_freq * harm)*np.cos(2*np.pi * t * fundamental_freq *harm) * A/harm

    return signal/n_harmonics, d_signal/n_harmonics, d2_signal/n_harmonics

def gen_stacks(n_samples,alpha=8.,theta=2.,sample_rate=44100,noise_sd=0.005):

    
    t = np.arange(0,0.1,1/sample_rate)
    #ff = 1650
    #alpha=8.
    #theta = 2.
    x = np.linspace(0,20,len(t))
    ff_fnc = x**(alpha-1)*np.exp(-x/theta)/(gamma(alpha) * theta**alpha)
    ff_fnc /= np.amax(ff_fnc)
    ff_fnc = 500 * ff_fnc + 1500

    s,d_true,d2_true = make_harmonic_stack(ff_fnc,n_harmonics=5,t=t)

    data=s + noise_sd * np.random.randn(n_samples,len(s))

    return data,d_true,d2_true

### Data from Coen Elemans & group

descriptions = {'3':'400Hz: (1) spike train, constant interval',
               '4.2':'400Hz: (3) no spike after 10ms',
               '4': '400Hz: (2) no spike 15-20ms',
               '5': '400Hz: (4) varied ISI: … 10, 11, 15… 25, 26, 30 … ',
               '6': '200Hz: (1) spike train, constant interval',
               '7': '200Hz: (2) no spike 15-20ms',
               '8': '200Hz: (3) varied ISI: … 10, 12, 20… 30, 32, 40 ',
               '9':'50Hz, spike train, constant interval'
               }
def load_coen_data(data_path,target_fs=0):

    ######### TO DO: ADD IN DOWNSAMPLING FOR THESE DATA #########

    audio_files = glob.glob(os.path.join(data_path,'UTF-8*.dat'))
    audio_files.sort()
    sim_nums = [a.split('_')[-1].split('.dat')[0] for a in audio_files]
    inputs = [os.path.join(data_path,s + '.inp') for s in sim_nums]
    flows = [os.path.join(data_path,f'qc_t1_0.5_1.1_0vs_{s}.dat') for s in sim_nums]
    descs = [descriptions[s] for s in sim_nums]
    ad = [np.genfromtxt(fn) for fn in audio_files]
    ts = [d[:,0]/1000 for d in ad]
    ys = [d[:,1] for d in ad]
    flws = [np.genfromtxt(fn) for fn in flows]
    spikes = [np.genfromtxt(fn,skip_header=2) for fn in inputs]
    srs = [int(round(1/(t[1] - t[0]))) for t in ts]
    if target_fs > 0:
        new_ts,new_ys = [],[]
        for t,y in zip(ts,ys):

            new_t = np.arange(t[0],t[-1],1/target_fs)
            yspl = make_interp_spline(t,y)
            new_y = yspl(new_t)
            new_ts.append(new_t)
            new_ys.append(new_y)
        return new_ts,new_ys,flws,spikes,target_fs,descs
    else:
        return ts,ys,flws,spikes,srs[0],descs

