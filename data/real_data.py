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
from data.preprocess import filter_by_tags

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

def get_all_audio(audio,fs,onOffs,context_len=0.02,max_pairs = 600,env=False,
                  current_total=0,full_vocs=False,extend=True,
                  padding=0.):

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

    #print(onOffs[:5,:])
    #print(f"padding by {padding} seconds")    
    onOffs[:,0] = np.maximum(onOffs[:,0]-padding,np.zeros(onOffs[:,0].shape))
    onOffs[:,1] = np.minimum(onOffs[:,1] +padding, len(audio)/fs *np.ones(onOffs[:,1].shape))
    #print(onOffs[:5,:])
    #assert False
        #onOffs = 
    
    for onset,offset in onOffs:

        aud = get_audio(audio,fs,onset,offset,context_len)
        #print(aud.shape)
        ts = np.arange(0,(len(aud)+1)/fs,1/fs)[:len(aud)]
        #spl = make_smoothing_spline(ts,aud)
        cut_len = np.mod(aud.shape[0],chunk_len)
        #print(cut_len)
        #print(aud[:minLen].shape)
        
        if aud.shape[0] >= chunk_len:
            if not full_vocs:
                if cut_len>0:
                    aud = aud[: - cut_len]

                aud = aud.reshape(-1,chunk_len,aud.shape[-1])

                auds.append([a[None,:,:] for a in aud])
            else:
                aud = aud.reshape(1,-1,1)

                auds.append(aud)
            
            ii += len(aud)
            print(f'current_total: {ii} samples',end='\r',flush=True)
            if ii >= max_pairs:
                break

    #print(f"kept {ii}/{total_vocs} vocalizations more than 20 ms long")
    return auds

def make_marmo_seg_file(matfile,savedir = ''):

    d = loadmat(matfile)
    vocal = d['vocal'][0][0]
    aud = vocal[0]
    L = len(aud)
    fs = vocal[1].squeeze()
    onset,offset = 0, L/fs
    voctype = vocal[5]
    savedir = ''.join(matfile.split('/')[:-1]) if savedir == '' else savedir
    fn = matfile.split('/')[-1].split('.mat')[0]    

    new_fn_wav = fn + '_' + voctype + '.wav'
    new_fn_seg = fn + '_' + voctype + '.txt'

    with open(new_fn_seg,'w') as f:
        f.write(str(onset) + '\t' + str(offset) + '\t' + str(voctype))

    wavfile.write(new_fn_wav,rate=fs,data=aud)

    return new_fn_wav,new_fn_seg


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

def get_segmented_audio(audiopath,segpath,audio_subdir='',seg_subdir='',\
                        max_pairs=5000,context_len=0.03,envelope=False,audio_type='.wav',
                        seg_type='.txt',seed=None,full_vocs=False,extend=True,
                        padding=0.,shuffle_order=True):
    
    """
    Takes as input a path to audio and segments (along with any
    shared subdirectories and file extensions),
    if used for gathering training data, outputs a list of
    1,L,1 audio chunks 
    if used for analysis, use the full_vocs option:
    outputs a list of lists, with each inner list containing the
    1,L_i,1 audio chunks (corresponding to vocalizations) within each audio file,
    which is treated as a separate session. 
        
    """

    #if audio_subdir[-1] != '/' and audio_subdir != '':
    #    audio_subdir += '/'
    #if seg_subdir[-1] != '/' and seg_subdir != '':
    #    seg_subdir += '/'
    random.seed(seed)
    audio_dirs = glob.glob(os.path.join(audiopath,audio_subdir))
    seg_dirs = glob.glob(os.path.join(segpath,seg_subdir))
    audio_dirs.sort()
    seg_dirs.sort()
    split_aud_sub = audio_subdir.split('/')
    split_seg_sub = seg_subdir.split('/')
    aud_sub_depth=0 if audio_subdir == '' else len(split_aud_sub)
    seg_sub_depth=0 if seg_subdir == '' else len(split_seg_sub)

    aud_sub_depth += 1
    seg_sub_depth += 1

    audio_tags = [a.split('/')[-aud_sub_depth] for a in audio_dirs]
    seg_tags = [s.split('/')[-seg_sub_depth] for s in seg_dirs]
    
    audio_dirs,seg_dirs  = filter_by_tags(audio_dirs,seg_dirs,audio_tags,seg_tags)
    assert len(audio_dirs) >0, \
        print(f"something went wrong with filtering! i recieved {audiopath},{segpath} as paths,{audio_subdir},{seg_subdir} as subdirs, found {audio_tags},{seg_tags} as tags")
    #print(audio_dirs,seg_dirs)
    #days = glob.glob(os.path.join(data_dir,'[0-9]*[0-9]'))
    #days = [d.split('/')[-1] for d in wav_dirs]
    #print("running this code")
    #assert False
    #print(f"searching for audio in {os.path.join(audiopath,'*' + audio_type)}")
    wavs = sum([glob.glob(os.path.join(a,'*' + audio_type)) for a in audio_dirs],[])
    wavs.sort()
    #print(f"searching for segments in {os.path.join(segpath,'*' + seg_type)}")
    segs = sum([glob.glob(os.path.join(s,'*' + seg_type)) for s in seg_dirs],[])
    segs.sort()
    audio_tags = [a.split(audio_type)[0].split('/')[-1] for a in wavs]
    seg_tags = [s.split(seg_type)[0].split('/')[-1] for s in segs]

    #print(audio_tags,seg_tags)
    #print(len(wavs))
    #print(len(segs))
    #print(wavs,segs)
    #print(audio_tags[:5],seg_tags[:5])
    wavs,segs = filter_by_tags(wavs,segs,audio_tags,seg_tags)
    #print(wavs[:5],segs[:5])
    assert len(wavs) > 0,\
        print(f"""
              something went wrong with filtering! i recieved {audiopath},{segpath} as paths,{audio_subdir},{seg_subdir} as subdirs, {audio_type},{seg_type} as filetypes.
              maybe something went wrong with the audio tags? here's what i received (first 5):
              audio tags: {audio_tags[:5]}
              segment tags: {seg_tags[:5]}
              """)
    #print(wavs,segs)
    """
    if len(wavs) != len(segs):
        print("different number of wavs and segs! only taking ones with overlap")
        wav_endings = [w.split('/')[-1].split(audio_type)[0] for w in wavs]
        seg_endings = [s.split('/')[-1].split(seg_type)[0] for s in segs]
        #print(wav_endings[:5])
        #print(seg_endings[:5])
        all_endings = set(wav_endings).intersection(seg_endings)
        wavs = [w for w in wavs if w.split('/')[-1].split(audio_type[-4:])[0].split('_cleaned')[0] in all_endings]
        segs = [s for s in segs if s.split('/')[-1].split(seg_type[-4:])[0] in all_endings]
    """
    #print(len(wavs))
    #print(len(segs))
    assert len(wavs) == len(segs), print(f"different number of wavs and segments: {len(wavs)} wavs and {len(segs)} segments")
    if shuffle_order:
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
                onoffs = np.loadtxt(v,usecols=(0,1))
            if len(onoffs) > 0:
                if len(onoffs.shape)==1:
                    onoffs = onoffs[None,:]
                if onoffs.shape[1] == 3:
                    onoffs = onoffs[:,:2]
                
                audios = get_all_audio(audio,sr,onoffs,max_pairs=max_pairs,\
                                       context_len=context_len,env=envelope,\
                                        current_total=current_total,full_vocs=full_vocs,
                                        extend=extend,padding=padding)
                #print(len(audios))
                if not full_vocs:
                    audio_segs += audios

                else:
                    audio_segs.append(audios)
                
                current_total += len(audios)
                #print(len(audio_segs))
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
