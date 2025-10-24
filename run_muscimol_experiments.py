import fire
from data.real_data import get_segmented_audio
from data.data_utils import get_loaders
from train.model_cv import model_cv_lambdas
from train.train import load_model
from train.eval import full_eval_model
import os
import numpy as np
import matplotlib.pyplot as plt
from visualization.model_vis import format_axes
import pickle
from analysis.analysis import get_budgie_fncs,get_marmo_fncs
import gc 
import glob
import torch

def run_experiments(audio_path='',
                    model_path='',seed=92,tau=1e-4,lr=1e-3,n_kernels=10,
                    max_sylls=3000):
    
    seg_path = os.path.join(audio_path,'segs')
    days = glob.glob(os.path.join(seg_path,'8[0-9]'))
    syllables = glob.glob(os.path.join(days[0],'syllable*'))
    n_sylls = len(syllables)
    max_per_syll = max_sylls//n_sylls
    audios = []
    for syll in syllables:
        syll = syll.split(days[0])[-1].split('/')[-1]
        #print(seg_path,syll)
        print(f"loading {max_per_syll} sylls from {syll}")
        seg_subdir = os.path.join('8[0-9]',syll) #syll.split(days[0])[-1]
        #print(seg_subdir)
        #print(max_per_syll)
        a,sr = get_segmented_audio(audio_path,seg_path,audio_subdir='8[0-9]/synchro_squeezed',\
                                    seg_subdir=seg_subdir,envelope=False,context_len=0.15,\
                                    audio_type='_cleaned.wav',seg_type='.txt',\
                                        max_pairs=max_per_syll,seed=seed)
        audios.append(np.vstack(a))
    dls = get_loaders(np.vstack(audios),cv = True,train_size=0.6,seed=seed)

    model = model_cv_lambdas(dls,1/sr,\
                                    nEpochs=200,model_path=model_path,
                                    n_layers=3,expand_factor=8,n_kernels=n_kernels,tau=tau,lr=lr)
    
    r2s,best,resids,spec_ratio,specs,ext = full_eval_model(model,dls,audios,1/sr,use_results=False,\
                    n_int=50,plot_dir=model_path,plot_steps=False)
    
    eval_dict={'r2s':r2s,
                    'best_data':best,
                    'resids':resids,
                    'spec_ratio':spec_ratio,
                    'specs':specs,
                    'ext':ext}
    
    with open(os.path.join(model_path,'eval_data.pkl'),'wb') as f:
        pickle.dump(eval_dict,f)


if __name__ == '__main__':

    fire.Fire(run_experiments)