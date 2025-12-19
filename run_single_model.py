from train.model_cv import model_cv_lambdas
from data.real_data import get_segmented_audio
from data.data_utils import get_loaders

import numpy as np
import fire 

def run_model(model_path='',audio_path='',seg_path='',audio_subdir='',seg_subdir='',
              audio_tag='_cleaned.wav',seg_tag='.txt',max_vocs=2000,
              context_len=0.15,nEpochs=200,seed=92,
              lr=1e-3,n_kernels=10
              ):
    """
    Docstring for run_model

    loads audio segments from audio_path and seg_path. Assumes that audio filenames
    match segment filenames other than extension at the end (determined by
    audio_tag, seg_tag)
    
    :param model_path: string, path to where models will be saved
    :param audio_path: string, path to parent directory containing audio
    :param seg_path: string, path to parent directory containing segments with audio
    :param audio_subdir: string, regular expression for subdirectories containing audio. 
                                will be added to end of seg_path
    :param seg_subdir: string, regular expression for subdirectories containing audio. 
                                will be added to end of seg_path
    :param audio_tag: string, extension that all audio files end with
    :param seg_tag: string, extension that all segment files end with
    :param max_vocs: int, max number of vocalizations to load
    :param context_len: float, length of samples (in seconds) that model will see during training
    :param nEpochs: number of full passes through training data
    :param seed: int or None; used for reproducibility
    :param lr: float, learning rate during training
    :param n_kernels: int, maxmimum degree of polynomials
    """
    
    a,sr = get_segmented_audio(audio_path,seg_path,audio_subdir=audio_subdir,\
                                seg_subdir=seg_subdir,envelope=False,context_len=context_len,\
                                audio_type=audio_tag,seg_type=seg_tag,\
                                    max_pairs=max_vocs,seed=seed)
    
    dls = get_loaders(np.vstack(a),cv=True,train_size=0.6,seed=seed)

    tau = 1/sr

    model = model_cv_lambdas(dls,1/sr,\
                                nEpochs=nEpochs,model_path=model_path,
                                n_layers=3,expand_factor=8,n_kernels=n_kernels,tau=tau,lr=lr)
    
if __name__ == '__main__':
    fire.Fire(run_model)
