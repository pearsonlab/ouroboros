import numpy as np
from tqdm import tqdm
import warnings

import os
import glob
from scipy.io import wavfile
import soundfile as sf
from scipy.io import loadmat
from utils import butter_filter
import random
from data.preprocess import filter_by_tags

from typing import Tuple


def get_audio(audio:np.ndarray,
               fs:float, onset:float, offset:float, context_len:float=0.3)->np.ndarray:
    """
    takes a full section of audio, extracts a short segment of it.

    inputs
    -----
        - audio: audio to extract
        - fs: audio sampling rate
        - onset: onset of segment to extract, in seconds
        - offset: offset of segment to extract, in seconds
        - context_len: minimum length of audio to extract

    returns
    -----
        segment of audio
    """

    
    difference = (offset - onset) - context_len
    if difference <= 0:
        # extend from the beginning, if shorter than context_len
        onset += difference  
    
    on = int(round(onset * fs))
    off = int(round(offset * fs))

    a = audio[on:off]

    return a[:, None]


def get_all_audio(
    audio:np.ndarray,
    fs:float,
    onOffs:np.ndarray,
    context_len:float=0.02,
    max_vocs:int=600,
    current_total:int=0,
    full_vocs:bool=False,
    extend:bool=True,
    padding:float=0.0,
)->list[np.ndarray]:
    """
    segments a long audio file into a set of shorter chunks, based on onsets and offsets from onOffs.
    has two modes: one that grabs full vocalizations, based on true onsets and offsets (full_vocs=True).
    This is used for evaluation.
    The default (full_vocs=False), grabs segments of length context_len. This is used for training.

    Inputs
    -----
        - audio: full audio from a wav file
        - fs: audio sample rate
        - onOffs: segments of audio from audio. These don't need to be discrete vocal units, but they should
        be segments that contain vocalizations
        - context_len: length of segmented audio in seconds, when full_vocs = False
        - max_vocs: maximum number of vocalizations to grab from the audio, total
        - current_total: number of vocalizations collected before this audio file
        - full_vocs: whether to grab full vocalizations (analysis) or short chunks (training)
        - extend: whether to extend full vocalizations to all match in length (samples)
        - padding: amount of padding to add to onsets and offsets, in seconds

    Returns
    -----
        - a list of collected audio
    """

    auds = []
    ii = current_total

    chunk_len = int(round(context_len * fs))

    if full_vocs and extend:
        # extend onoffs in a sensible way -- maybe to length of max onoff

        lens = onOffs[:, 1] - onOffs[:, 0]
        max_len = np.amax(lens)  # or something different here like mean, median, etc
        diffs = max_len - lens
        onOffs[:, 1] += diffs


    onOffs[:, 0] = np.maximum(onOffs[:, 0] - padding, np.zeros(onOffs[:, 0].shape))
    onOffs[:, 1] = np.minimum(
        onOffs[:, 1] + padding, len(audio) / fs * np.ones(onOffs[:, 1].shape)
    )


    for onset, offset in onOffs:
        aud = get_audio(audio, fs, onset, offset, context_len)

        cut_len = np.mod(aud.shape[0], chunk_len)


        if aud.shape[0] >= chunk_len:
            if not full_vocs:
                if cut_len > 0:
                    aud = aud[:-cut_len]

                aud = aud.reshape(-1, chunk_len, aud.shape[-1])

                for a in aud:
                    auds.append(a[None, :, :])
                
            else:
                aud = aud.reshape(1, -1, 1)

                auds.append(aud)

            ii += len(aud)
            print(f"current_total: {ii} samples", end="\r", flush=True)
            if ii >= max_vocs:
                break

    return auds


def make_marmo_seg_file(matfile:str, savedir:str="")->Tuple[str,str]:
    """
    takes a marmoset audio file (in `.mat` format), converts to a `.wav` (audio) and
    `.txt` (onset offset) file

    Inputs
    -----
        - matfile: location of `.mat` file
        - savedir: location to save the new file
    Returns
    -----
        - new_fn_wav: location of `.wav` file
        - new_fn_set: location of `.seg` file
    """

    d = loadmat(matfile)
    vocal = d["vocal"][0][0]
    aud = vocal[0]
    L = len(aud)
    fs = vocal[1].squeeze()
    onset, offset = 0, L / fs
    voctype = vocal[5]
    savedir = "".join(matfile.split("/")[:-1]) if savedir == "" else savedir
    fn = matfile.split("/")[-1].split(".mat")[0]

    new_fn_wav = fn + "_" + voctype + ".wav"
    new_fn_seg = fn + "_" + voctype + ".txt"

    with open(new_fn_seg, "w") as f:
        f.write(str(onset) + "\t" + str(offset) + "\t" + str(voctype))

    wavfile.write(new_fn_wav, rate=fs, data=aud)

    return new_fn_wav, new_fn_seg

def get_segmented_audio(
    audiopath,
    segpath,
    audio_subdir="",
    seg_subdir="",
    max_pairs=5000,
    context_len=0.03,
    envelope=False,
    audio_type=".wav",
    seg_type=".txt",
    seed=None,
    full_vocs=False,
    extend=True,
    padding=0.0,
    shuffle_order=True,
):
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

    # if audio_subdir[-1] != '/' and audio_subdir != '':
    #    audio_subdir += '/'
    # if seg_subdir[-1] != '/' and seg_subdir != '':
    #    seg_subdir += '/'
    random.seed(seed)
    audio_dirs = glob.glob(os.path.join(audiopath, audio_subdir))
    seg_dirs = glob.glob(os.path.join(segpath, seg_subdir))
    audio_dirs.sort()
    seg_dirs.sort()
    split_aud_sub = audio_subdir.split("/")
    split_seg_sub = seg_subdir.split("/")
    aud_sub_depth = 0 if audio_subdir == "" else len(split_aud_sub)
    seg_sub_depth = 0 if seg_subdir == "" else len(split_seg_sub)

    aud_sub_depth += 1
    seg_sub_depth += 1

    audio_tags = [a.split("/")[-aud_sub_depth] for a in audio_dirs]
    seg_tags = [s.split("/")[-seg_sub_depth] for s in seg_dirs]

    audio_dirs, seg_dirs = filter_by_tags(audio_dirs, seg_dirs, audio_tags, seg_tags)
    assert len(audio_dirs) > 0, print(
        f"something went wrong with filtering! i recieved {audiopath},{segpath} as paths,{audio_subdir},{seg_subdir} as subdirs, found {audio_tags},{seg_tags} as tags"
    )
    # print(audio_dirs,seg_dirs)
    # days = glob.glob(os.path.join(data_dir,'[0-9]*[0-9]'))
    # days = [d.split('/')[-1] for d in wav_dirs]
    # print("running this code")
    # assert False
    # print(f"searching for audio in {os.path.join(audiopath,'*' + audio_type)}")
    wavs = sum([glob.glob(os.path.join(a, "*" + audio_type)) for a in audio_dirs], [])
    wavs.sort()
    # print(f"searching for segments in {os.path.join(segpath,'*' + seg_type)}")
    segs = sum([glob.glob(os.path.join(s, "*" + seg_type)) for s in seg_dirs], [])
    segs.sort()
    audio_tags = [a.split(audio_type)[0].split("/")[-1] for a in wavs]
    seg_tags = [s.split(seg_type)[0].split("/")[-1] for s in segs]

    # print(audio_tags,seg_tags)
    # print(len(wavs))
    # print(len(segs))
    # print(wavs,segs)
    # print(audio_tags[:5],seg_tags[:5])
    wavs, segs = filter_by_tags(wavs, segs, audio_tags, seg_tags)
    # print(wavs[:5],segs[:5])
    assert len(wavs) > 0, print(f"""
              something went wrong with filtering! i recieved {audiopath},{segpath} as paths,{audio_subdir},{seg_subdir} as subdirs, {audio_type},{seg_type} as filetypes.
              maybe something went wrong with the audio tags? here's what i received (first 5):
              audio tags: {audio_tags[:5]}
              segment tags: {seg_tags[:5]}
              """)
    # print(wavs,segs)
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
    # print(len(wavs))
    # print(len(segs))
    assert len(wavs) == len(segs), print(
        f"different number of wavs and segments: {len(wavs)} wavs and {len(segs)} segments"
    )
    if shuffle_order:
        order = np.random.choice(len(wavs), len(wavs), replace=False)
        wavs = [wavs[o] for o in order]
        segs = [segs[o] for o in order]
    audio_segs = []
    # print(f'number of wavs: {len(wavs)}')
    # print(f'number of segs: {len(segs)}')
    current_total = 0
    # print(len(wavs),len(segs))
    # print(max_pairs)
    if ".mat" not in audio_type:
        for ii, (w, v) in enumerate(zip(wavs, segs)):
            if ".wav" in audio_type:
                sr, audio = wavfile.read(w)
            elif ".flac" in audio_type:
                # print(f"file number {ii+1}")
                audio, sr = sf.read(w)
            if audio.dtype == np.int16:
                audio = audio / -np.iinfo(audio.dtype).min

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                onoffs = np.loadtxt(v, usecols=(0, 1))
            if len(onoffs) > 0:
                if len(onoffs.shape) == 1:
                    onoffs = onoffs[None, :]
                if onoffs.shape[1] == 3:
                    onoffs = onoffs[:, :2]

                audios = get_all_audio(
                    audio,
                    sr,
                    onoffs,
                    max_pairs=max_pairs,
                    context_len=context_len,
                    env=envelope,
                    current_total=current_total,
                    full_vocs=full_vocs,
                    extend=extend,
                    padding=padding,
                )
                # print(len(audios))
                # print(len(audios))
                if not full_vocs:
                    audio_segs += audios

                else:
                    audio_segs.append(audios)

                current_total += len(audios)
                # print(current_total)
                # print(len(audio_segs))
                # assert len(audio_segs) >= current_total,print("wtf")
                if current_total >= max_pairs:
                    return audio_segs[:max_pairs], sr
    else:
        random.shuffle(wavs)
        for w in tqdm(wavs, desc="Getting audio from .mat files"):
            audio, sr = get_audio_from_mat(w, context_len=context_len, env=envelope)
            audio_segs += audio
            if len(audio_segs) >= max_pairs:
                return audio_segs[:max_pairs], sr

    return audio_segs, sr


def grab_segments(seg_list, *args, fs=42000):

    #### IMPLEMENT
    output = [[] for _ in args]
    for ii, onoffs in enumerate(seg_list):
        # onoffs = np.loadtxt(seg_file)
        if np.ndim(onoffs) == 1:
            onoffs = onoffs[None, :]
        for onInd, offInd in onoffs:
            # onInd, offInd = int(round(on*fs)),int(round(off*fs))
            for jj, inputs in enumerate(args):
                # print(inputs[ii].squeeze()[onInd:offInd].shape)
                output[jj].append(inputs[ii][onInd:offInd])

        # print(len(output[0]))
        # assert False
        # print(output)
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
