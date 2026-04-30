from train.model_cv import model_cv_lambdas
from data.real_data import get_segmented_audio
from data.data_utils import get_loaders,aud_neur_ds
import glob
import os
import numpy as np
import fire 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

def run_model(model_path='',audio_path='',marmo_regex='',
              audio_tag='_cleaned.wav',seg_tag='.txt',max_per_marmo=500,
              context_len=0.15,nEpochs=200,seed=92,
              lr=1e-3,n_kernels=10,batch_size=32,
              tau = 1e-3
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
    num_workers = os.cpu_count()-1
    marmos = glob.glob(os.path.join(audio_path,marmo_regex))

    x_train_all,x_test_all,x_val_all = [],[],[]
    dls = {}
    for m in marmos:
        print(f"getting audio for {m.split('/')[-1]}")
        a,sr = get_segmented_audio(m,m,audio_subdir='vocs/s*/wavfiles/double_denoised',\
                                seg_subdir='vocs/s*/wavfiles',envelope=False,context_len=context_len,\
                                audio_type=audio_tag,seg_type=seg_tag,\
                                    max_pairs=max_per_marmo,seed=seed)
        
        X_train,X_test = train_test_split(np.vstack(a),test_size=0.4,random_state=seed)
        X_val, X_test = train_test_split(X_test,test_size=0.5,random_state=seed)
        x_train_all.append(X_train)
        x_test_all.append(X_test)
        x_val_all.append(X_val)

    dsVal = aud_neur_ds(np.vstack(x_val_all))
    dsTrain,dsTest = aud_neur_ds(np.vstack(x_train_all)),aud_neur_ds(np.vstack(x_test_all))
    dls['val'] = DataLoader(dsVal,num_workers=num_workers,batch_size=batch_size,shuffle=False)
    dls['train'] = DataLoader(dsTrain,num_workers=num_workers,batch_size=batch_size,shuffle=True)
    dls['test'] = DataLoader(dsTest,num_workers=num_workers,batch_size=batch_size,shuffle=False)


    #dls = get_loaders(np.vstack(all_audio),cv=True,train_size=0.6,seed=seed,batch_size=batch_size,dt=1/sr)
    #print(np.vstack(a).shape)
    print(f"tau = {tau}")
    if tau > 1/sr:
        tau = 1/sr
    #tau = 1/sr
    print(f"tau = {tau}")

    model = model_cv_lambdas(dls,1/sr,\
                                nEpochs=nEpochs,model_path=model_path,
                                n_layers=3,expand_factor=8,n_kernels=n_kernels,tau=tau,lr=lr)
    
if __name__ == '__main__':
    fire.Fire(run_model)
