import numpy as np
import warnings

import os
import glob
from scipy.io import wavfile
import soundfile as sf
from scipy.io import loadmat
import random
from data.preprocess import filter_by_tags

from typing import Tuple, Union

def get_audio_training(audio_files:list[str],
                       seg_files:list[str],
                       max_vocs:int = 5000,
                       context_len:float=0.3,
                       extend:bool=True):

    total_vocs = 0
    audio=[]
    for a, s in zip(audio_files,seg_files):
        if os.path.isfile(a) and os.path.isfile(s):
            sr,aud = wavfile.read(a)

            seg_onoffs = np.loadtxt(s)
            if len
    pass

def get_audio_analysis():

    pass

def get_segmented_audio(
    audiopath: str,
    segpath: str,
    audio_id: str = '.wav',
    max_vocs: int = 5000,
    context_len: float = 0.3,
    seed: Union[None, int] = None,
    training: bool = False,
    extend: bool = True,
    padding: float = 0.0,
    shuffle_order: bool = True
) -> Tuple[list,int]:
    
    gen = np.random.default_rng(seed=seed)

    audio_files = glob.glob(os.path.join(audiopath,'*'+audio_id))
    audio_tags = [a.split('/')[-1].split(audio_id)[0] for a in audio_files]

    seg_files = [os.path.join(segpath,a) for a in audio_tags]

    if shuffle_order:
        order = gen.choice(len(audio_files),len(audio_files),replace=False)
        audio_files = [audio_files[o] for o in order]
        seg_files = [seg_files[o] for o in order]

    if training:

        audio_segments, sr = get_audio_training(audio_files,
                                                seg_files,
                                                max_vocs=max_vocs,
                                                context_len=context_len,
                                                extend=extend)

    else:
        audio_segments,sr = get_audio_analysis(audio_files,
                                                seg_files,
                                                max_vocs=max_vocs,
                                                padding=padding)

    return audio_segments,sr