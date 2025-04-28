import numpy as np
from tqdm import tqdm
import warnings

from scipy.signal import hilbert
from scipy.interpolate import make_smoothing_spline
#from acoustic_features_torch import get_features
import os
import glob
from scipy.io import wavfile 
import soundfile as sf
from scipy.io import loadmat
from utils import butter_filter
import random

def get_audio(audio,fs,onset,offset,context_len=0.3):

    #audiotimes = np.linspace(0,len(audio)/fs,len(audio))
    difference = (offset - onset) - context_len
    if difference <= 0:
        onset += difference #context_len//2
    #on,off = np.searchsorted(audiotimes,onset),np.searchsorted(audiotimes,offset)
    on = int(round(onset * fs))
    off = int(round(offset * fs))
    
    a = audio[on:off]

    #print(a.shape)
    return a[:,None]

def get_all_audio(audio,fs,onOffs,context_len=0.02,max_pairs = 600,env=False,current_total=0,full_vocs=False,extend=True):

    spikes = []
    auds = []
    ii = current_total
    total_vocs = len(onOffs)
    

    chunk_len = int(round(context_len * fs))
    if env:
        analytic_signal = hilbert(audio)
        envelope=np.abs(analytic_signal)
    else:
        envelope=[]

    if full_vocs and extend:
        # extend onoffs in a sensible way -- maybe to length of max onoff
        
        lens = onOffs[:,1] - onOffs[:,0]
        max_len = np.amax(lens) # or something different here like mean, median, etc
        diffs = max_len - lens 
        onOffs[:,1] += diffs
        

        #onOffs = 
    
    for onset,offset in onOffs:

        aud = get_audio(audio,fs,onset,offset,context_len)
        ts = np.arange(0,(len(aud)+1)/fs,1/fs)[:len(aud)]
        #spl = make_smoothing_spline(ts,aud)
        cut_len = np.mod(aud.shape[0],chunk_len)
        #print(aud[:minLen].shape)
        
        if aud.shape[0] >= chunk_len:
            if not full_vocs:
                aud = aud[: - cut_len]

                aud = aud.reshape(-1,chunk_len,aud.shape[-1])
            else:
                aud = aud.reshape(1,-1,1)

            auds.append(aud)
            
            ii += len(aud)
            print(f'current_total: {ii} samples',end='\r',flush=True)
            if ii >= max_pairs:
                break

    #print(f"kept {ii}/{total_vocs} vocalizations more than 20 ms long")
    return auds

def get_audio_from_mat(matfile,context_len=0.02,max_pairs=600,env=False):

    d = loadmat(matfile)
    vocal = d['vocal'][0][0]
    data = vocal[0]
    if data.dtype == np.int16:
        data = data/-np.iinfo(data.dtype).min
    fs = vocal[1].squeeze()
    dt = 1/fs
    chunk_len = int(round(context_len * fs))

    data = butter_filter(data.squeeze(),cutoff=600,order=5,fs=fs,btype='high')[:,None]
    cut_len = np.mod(len(data),chunk_len)
    if len(data) >= chunk_len:
        if cut_len > 0:
            data = data[:-cut_len]

        data = data.reshape(-1,chunk_len,1)
        return [data],fs
    else:
        return [],fs
    
def get_sylltype_from_mat(matfiles,max_vocs=500,voctype='trill'):

    random.shuffle(matfiles)
    vocal_data = []
    for f in matfiles:
        d = loadmat(f)['vocal'][0][0] 
        dvoc = d[5]
        if dvoc == voctype:
            vocal_data.append(d[0])

        if len(vocal_data) >= max_vocs:
            break

    return vocal_data, d[1]

def get_segmented_audio(audiopath,segpath,max_pairs=5000,context_len=0.03,envelope=False,audio_type='.wav',
                        seg_type='.txt',seed=None,full_vocs=False,extend=True):

    random.seed(seed)
    #days = glob.glob(os.path.join(data_dir,'[0-9]*[0-9]'))
    #days = [d.split('/')[-1] for d in wav_dirs]
    #print("running this code")
    #assert False
    #print(f"searching for audio in {os.path.join(audiopath,'*' + audio_type)}")
    wavs = glob.glob(os.path.join(audiopath,'*' + audio_type))
    wavs.sort()
    #print(f"searching for segments in {os.path.join(segpath,'*' + seg_type)}")
    segs = glob.glob(os.path.join(segpath,'*' + seg_type))
    segs.sort()
    print(len(wavs))
    print(len(segs))
    #print(wavs,segs)
    if len(wavs) != len(segs):
        print("different number of wavs and segs! only taking ones with overlap")
        wav_endings = [w.split('/')[-1].split(audio_type[-4:])[0] for w in wavs]
        seg_endings = [s.split('/')[-1].split(seg_type[-4:])[0] for s in segs]
        all_endings = set(wav_endings).intersection(seg_endings)
        wavs = [w for w in wavs if w.split('/')[-1].split(audio_type[-4:])[0] in all_endings]
        segs = [s for s in segs if s.split('/')[-1].split(seg_type[-4:])[0] in all_endings]

    print(len(wavs))
    print(len(segs))
    assert len(wavs) == len(segs), print(f"different number of wavs and segments: {len(wavs)} wavs and {len(segs)} segments")
    order = np.random.choice(len(wavs),len(wavs),replace=False)
    wavs = [wavs[o] for o in order]
    segs = [segs[o] for o in order]
    audio_segs = []
    #print(f'number of wavs: {len(wavs)}')
    #print(f'number of segs: {len(segs)}')
    current_total=0
    if '.mat' not in audio_type:
        for ii,(w,v) in enumerate(zip(wavs,segs)):
            
            if '.wav' in audio_type:
                sr,audio = wavfile.read(w)
            elif '.flac' in audio_type:
                #print(f"file number {ii+1}")
                audio,sr = sf.read(w)
            if audio.dtype == np.int16:
                audio = audio/-np.iinfo(audio.dtype).min

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                onoffs = np.loadtxt(v)
            if len(onoffs) > 0:
                if len(onoffs.shape)==1:
                    onoffs = onoffs[None,:]
                if onoffs.shape[1] == 3:
                    onoffs = onoffs[:,:2]
                
                audios = get_all_audio(audio,sr,onoffs,max_pairs=max_pairs,\
                                       context_len=context_len,env=envelope,\
                                        current_total=current_total,full_vocs=full_vocs,extend=extend)
                if not full_vocs:
                    audio_segs += audios

                else:
                    audio_segs.append(audios)
                
                current_total += len(audios)
                #assert len(audio_segs) >= current_total,print("wtf")
                if current_total >= max_pairs:
                    return audio_segs[:max_pairs],sr
    else:
        random.shuffle(wavs)
        for w in tqdm(wavs,desc='Getting audio from .mat files'):
            audio, sr = get_audio_from_mat(w,context_len=context_len,env=envelope)
            audio_segs += audio 
            if len(audio_segs) >= max_pairs:
                return audio_segs[:max_pairs],sr
    
    return audio_segs,sr


def grab_segments(seg_list,*args,fs=42000):

    #### IMPLEMENT
    output = [[] for _ in args]
    for ii, onoffs in enumerate(seg_list):
        #onoffs = np.loadtxt(seg_file)
        if np.ndim(onoffs) == 1:
            onoffs = onoffs[None,:]
        for onInd,offInd in onoffs:
            #onInd, offInd = int(round(on*fs)),int(round(off*fs))
            for jj,inputs in enumerate(args):
                #print(inputs[ii].squeeze()[onInd:offInd].shape)
                output[jj].append(inputs[ii][onInd:offInd])

        #print(len(output[0]))
        #assert False
        #print(output)
    return output

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
