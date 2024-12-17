import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader


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
