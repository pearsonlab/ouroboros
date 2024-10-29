import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader


def gen_pure_tones(n_samples=8000,sample_rate=44100,frequency=5000,sample_length=0.3):


    samples = []

    gen = np.random.default_rng()

    for ii in range(n_samples):

        start_time = gen.uniform(low=0,high=1)

        t = np.arange(start_time,start_time + sample_length,1/sample_rate)

        y = np.sin(2*np.pi * frequency * t)

        samples.append(y)

    return np.stack(samples,axis=0)

