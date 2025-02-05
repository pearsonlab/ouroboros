import numpy as np
from tqdm import tqdm
import warnings

from scipy.signal import hilbert
#from acoustic_features_torch import get_features
import os
import glob
from scipy.io import wavfile 

def get_audio(audio,fs,onset,offset,env=False,envelope = []):

    audiotimes = np.linspace(0,len(audio)/fs,len(audio))
    on,off = np.searchsorted(audiotimes,onset),np.searchsorted(audiotimes,offset)
    a,t = audio[on:off],\
        audiotimes[on:off]
    if env:
        assert len(envelope) == len(audio)

        e = envelope[on:off]
        a = np.vstack([e,a]).T
    else:
        a = a[:,None]

    #print(a.shape)
    return a,t

def get_all_audio(audio,fs,onOffs,context_len=0.02,max_pairs = 600,env=False):

    spikes = []
    auds = []
    ii = 0
    total_vocs = len(onOffs)
    

    chunk_len = int(round(context_len * fs))
    analytic_signal = hilbert(audio)
    envelope=np.abs(analytic_signal)
    
    for onset,offset in onOffs:

        aud,spec_times = get_audio(audio,fs,onset,offset,env=env,envelope=envelope)
        
        cut_len = np.mod(aud.shape[0],chunk_len)
        #print(aud[:minLen].shape)
        
        if aud.shape[0] >= chunk_len:
            
            aud = aud[: - cut_len]

            aud = aud.reshape(-1,chunk_len,aud.shape[-1])

            auds.append(aud)
            
            ii += 1
            if ii >= max_pairs:
                break

    #print(f"kept {ii}/{total_vocs} vocalizations more than 20 ms long")
    return auds

def get_segmented_audio(audiopath,segpath,max_pairs=5000,context_len=0.03,envelope=False,audio_type='.wav',
                        seg_type='.txt'):

    #days = glob.glob(os.path.join(data_dir,'[0-9]*[0-9]'))
    #days = [d.split('/')[-1] for d in wav_dirs]
    wavs = glob.glob(os.path.join(audiopath,'*.wav'))
    wavs.sort()
    segs = glob.glob(os.path.join(segpath,'*.txt'))
    segs.sort()

    audio_segs = []
    for w,v in tqdm(zip(wavs,segs),desc='Getting audio from wav files'):

        sr,audio = wavfile.read(w)
        if audio.dtype == np.int16:
            audio = audio/-np.iinfo(audio.dtype).min

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            onoffs = np.loadtxt(v)
        if len(onoffs) > 0:
            if len(onoffs.shape)==1:
                onoffs = onoffs[None,:]
            if len(onoffs.shape) == 3:
                onoffs = onoffs[:,:2]
            #onoffs = np.hstack([onoffs,np.ones((onoffs.shape[0],1))])
            #print(onoffs.shape)
            audios = get_all_audio(audio,sr,onoffs,max_pairs=max_pairs,context_len=context_len,env=envelope)
    
            audio_segs += audios
            if len(audio_segs) >= max_pairs:
                return audio_segs[:max_pairs]

    return audio_segs,sr

"""
def get_feature_musd_dataset(samplerate,spiketimes,audio,onOffs,shoulder=0.01,data_type='rates'):

    xi2,xi,N = np.zeros((6,)),np.zeros((6,)),0
    for onset,offset,vocID in tqdm(onOffs,desc = "getting mean, sd of all audio"):

        aud,spike,*_ = get_audio_rate_pairs(audio,samplerate,spiketimes,onset,offset,data_type=data_type)

        #minLen = min(spike.shape[-1],aud.shape[-1])
        #inData = from_numpy(np.concatenate([spike[:,:minLen],aud[None,:minLen]],axis=0)).T[None,:,:]
        #if mask_audio:
        #    mask = torch.ones(inData.shape,device=inData.device)
        #    mask[:,:,-1] = 0
        #    inData *= mask

        #preds = (model(inData) + inData)[:,:,-1]
        print(aud.shape)
        features= get_features(from_numpy(aud)[None,:],samplerate=samplerate) ## needs to output time x feat num

        xi2 += (features**2).sum(axis=0)
        xi += features.sum(axis=0)
        N += features.shape[0]
        #print(features.shape)
        #assert False

        
    mu = xi/N
    sd = xi2/N - mu**2

    return mu,sd
"""
