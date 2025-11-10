from ssqueezepy import ssq_cwt,cwt,issq_cwt,ssq_stft,stft
from ssqueezepy.visuals import imshow, plot, scat
#from ssqueezepy.toolkit import lin_band
from scipy.io import wavfile
import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt
import soundfile as sf
import glob
import os
from itertools import repeat
from joblib import Parallel,delayed
from utils import butter_filter
import noisereduce as nr

### wavelets:
#morlet: higher mu -> greater frequency, lesser time resolution. for mu > 6
# wavelet almost exactly gaussian. mu=13.4: matches generalized morse wavelets (3,60)
# generalized morse wavelet (gmw): options are time & frequency spread
# bump: wider variance in time, narrower variance in frequency
#cmhat: complex mexican hat. wider frequency variance, narrower time variance
#hhhat: hilbert analytic function of hermitian hat. no idea what this does, don't use it
#  
HP_DICT = {
    'chunk length': 10000,
    'wavelet': 'morlet', #options are gmw, morlet, bump, cmhat,hhhat
    'band min': 1000.,
    'band max': 10000.,
    'nv': 32, #number of voices (wavelets per octave),
    'scales': 'log-piecewise',
    'order':5, # polynomial order for band-pass filter,
    'prop_reduce':1, # proportion of noise to reduce, if reducing noise
    'squeeze_freqs':True
}

WAVELET_HP_DICT = {
    'morlet': {'mu':13.4},
    'bump': {'mu': 5,'s':1,'om':0},
    'cmhat': {'mu':1,'s':1},
    'hhhat':{'mu':5},
    'gmw':{'gamma':6,'beta':60,}    
}

FILTER_DICT = {
    'chunk length': 10000,
    'band min': 1000.,
    'band max': 10000.
}

def viz(x, Tx, Wx,vmin=None,vmax=None,axs=True):
    ax = plt.gca()
    if vmin != None and vmax != None:
        plt.imshow(np.abs(Wx), aspect='auto', cmap='turbo',vmin=vmin,vmax=vmax)
    else:
        plt.imshow(np.abs(Wx), aspect='auto', cmap='turbo')
    if not axs:
        print("removing ticks")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def lin_band(Tx,slope,offset,bw,show=True,**kw):

    na, N = Tx.shape
    tcs = np.linspace(0, 1, N)
    Cs       = (slope*tcs + offset) * na
    freqband = bw * na * np.ones(N)
    Cs, freqband = Cs.astype('int32'), freqband.astype('int32')
    #print(Cs[0],freqband[0])
    #print("")
    if show:
        imshow(Tx, abs=1, aspect='auto', show=0, **kw)
        plot(Cs + freqband, color='r')
        plot(Cs - freqband, color='r', show=1)
    return Cs, freqband

def band_pass_preprocess(data,chunk_len,low_cut,high_cut,fs,kw,tn,return_full_ssq=True,order=5,show=True):

    print("using band-pass preprocessing")

    full_rec = []
    chunk_ons = np.arange(0,len(data),chunk_len)
    for ii,on in enumerate(chunk_ons):
        off = min(len(data),on + chunk_len)
        filtered = butter_filter(data[on:off],np.array([low_cut,high_cut]),fs,order=order,btype='band')
        full_rec.append(filtered)
    full_rec = np.hstack(full_rec)
    if return_full_ssq:
        #print(full_rec.shape)
        
        full_ssq,_,full_freqs,full_scales,*_ = ssq_cwt(data,t=tn,**kw)
        return full_rec,full_freqs,full_ssq
    return full_rec,None,None

def ssq_preprocess(data,tn,kw,chunk_len,show=True,min_band=0.01,max_band=0.35,return_full_ssq=True):
    
   
    if show:
        pass
        #_Tn = np.pad(Tn, [[4, 4]])  # improve display of top- & bottom-most freqs
        #imshow(Wn, **pkw)
        #imshow(_Tn, norm=(0, 4e-1), **pkw)

    full_rec = []
    slopen = -0.00 #flat band across time


    chunk_ons = np.arange(0,len(data),chunk_len)
    #print(data.shape,tn.shape)
    
    for ii,on in enumerate(chunk_ons):
        #print(f"running chunk {ii+1}/{len(chunk_ons)}",end='\r')
        off = min(len(data),on + chunk_len)
        nrec = np.zeros(data.shape)
        Tn, _,ssq_freqs,scales,*_ = ssq_cwt(data[on:off], t=tn[on:off], **kw)
        if max_band > 1:
            #print(max_band,min_band)
            nf = Tn.shape[0]

            max_band_new = np.sum(ssq_freqs >= min_band)/nf
            min_band_new = np.sum(ssq_freqs >= max_band)/nf
            
            min_band = min_band_new
            max_band = max_band_new
            #assert False
        bwn = (max_band - min_band)/2
        offsetn = (min_band + max_band)/2
        Csn, freqbandn = lin_band(Tn, slopen, offsetn, bwn, norm=(0, 4e-1),show=show)
        nrec = issq_cwt(Tn, kw['wavelet'], Csn, freqbandn)[0]
        full_rec.append(nrec)

    full_rec = np.hstack(full_rec)
    print("",end='\r',flush=True)
    if show:
        ax = plt.gca()
        ax.plot(data[1200:1400],label='data')
        ax.plot(full_rec[1200:1400], alpha=0.25,label='reconstruction')
        plt.legend()
        plt.show()
        plt.close()
        Tsxo, Sxo, *_ = ssq_stft(data)
        viz(data, np.flipud(Tsxo), np.flipud(Sxo),vmin = np.amin(np.abs(Sxo)),vmax=np.amax(np.abs(Sxo)))
        Tsx, Sx, *_ = ssq_stft(full_rec)
        viz(full_rec, np.flipud(Tsx), np.flipud(Sx),vmin = np.amin(np.abs(Sxo)),vmax=np.amax(np.abs(Sxo)))
    if return_full_ssq:
        #print(full_rec.shape)
        
        full_ssq,_,full_freqs,full_scales,*_ = ssq_cwt(data,t=tn,**kw)
        #print(full_freqs[-1],full_freqs[0],ssq_freqs[-1],ssq_freqs[0])
        #assert full_ssq.shape[0] == Tn.shape[0]
        return full_rec,full_freqs,full_ssq
    else:
        return full_rec,ssq_freqs,None

def check_valid(data,dtype,default=''):

    if data == default:

        return data,True 
    try:
        d = dtype(data)
        return d, True
    except:
        return data, False
    
def _tune_input_helper(p):
    """Get parameter adjustments from the user."""
    for key in p.keys():
        curr_dtype = type(p[key])
        #temp = 'not (number or empty)'
        temp = input('Set value for '+key+': ['+str(p[key])+ '] ')
        temp,valid = check_valid(temp,curr_dtype)
        while not valid:
            temp = input('Set value for '+key+': ['+str(p[key])+ '] ')
            temp,valid = check_valid(temp,curr_dtype)
        if temp != '':
            p[key] = temp
    return p

def _tuning_plot(orig_spec,cleaned_spec,ts,spec_freqs,scale_freqs,\
                 scaleogram,min_band,max_band,vmin=-0.5,vmax=1.5,\
                    save_loc='./pp.pdf'):

    
    nf,T = scaleogram.shape
    #assert len(scale_freqs) == ns, print(ns,len(scale_freqs))
    #scale_extent[2] = 0
    #scale_extent[3] = ns
    if max_band > 1:
        
        max_band_new = np.sum(scale_freqs >= min_band)/nf
        min_band_new = np.sum(scale_freqs >= max_band)/nf
        
        min_band = min_band_new
        max_band = max_band_new
    bw = (max_band - min_band)/2
    offset = (max_band + min_band)/2
    #print(offset,bw)
    
    band = bw * np.ones(T) *nf
    Cs = offset*nf *np.ones(T)
    #print(Cs[0])
    #print(band[0])

    vmin_scale = np.amin(scaleogram)
    vmax_scale = np.amax(scaleogram)
    vmax_scale = (vmax_scale - vmin_scale)/50 + vmin_scale
    fig,axs = plt.subplots(nrows=1,ncols=3,figsize=(13,4))
    spec_extent = [ts[0],ts[-1],spec_freqs[0],spec_freqs[-1]]
    scale_extent = [ts[0],ts[-1],nf,0]
    axs[0].imshow(orig_spec,origin='lower',aspect='auto',vmin=vmin,vmax=vmax,extent=spec_extent)
    axs[1].imshow(cleaned_spec,origin='lower',aspect='auto',vmin=vmin,vmax=vmax,extent=spec_extent)
    axs[2].imshow(scaleogram,aspect='auto',cmap='bone',vmin=vmin_scale,vmax=vmax_scale,extent=scale_extent)
    #axs[2].set_xticks(np.arange(0,len(ts),len(ts)/10),ts[::int(round(len(ts)/10))])
    axs[2].set_yticks(np.arange(0,nf,25),np.round(scale_freqs[::25]).astype(np.int32))
    axs[2].plot(ts,Cs - band,color='r')
    axs[2].plot(ts,Cs + band,color='r')
    axs[0].set_ylabel("Frequency (Hz)")
    axs[0].set_title("Original spectrogram")
    axs[1].set_title("Reconstructed spectrogram")
    axs[2].set_title("CWT with reconstruction boundaries")
    for ax in axs:
        ax.set_xlabel("Time (s)")
    plt.tight_layout
    plt.savefig(save_loc)
    plt.close('all')


def tune_preprocessing(audio_files,segment_files,hp_dict,preprocess_type='ssq',img_fn='./pp.pdf',reduce_noise=True):

    """
    tune preprocessing parameters for denoising spectrograms
    Parameters
    ----------
    audio_files : list of str
        Audio files
    seg_files : list of str
        Segment files
    p : dict
        Preprocessing parameters: Add a reference!

    Returns
    -------
    p : dict
        Adjusted preprocessing parameters.

    """

    assert len(audio_files) == len(segment_files), print("number of audio files is not same as number of segments!")
    
    while True:


        p = _tune_input_helper(hp_dict)
        resp = 'nothing yet'
        while resp != 's' and resp != 'r':
            ind = np.random.choice(len(audio_files))
            audio_fn = audio_files[ind]
            seg_fn = segment_files[ind]
            print(audio_fn,seg_fn)

            onoffs = np.loadtxt(seg_fn,usecols=(0,1))
            while len(onoffs) == 0:
                ind = np.random.choice(len(audio_files))
                audio_fn = audio_files[ind]
                seg_fn = segment_files[ind]
                print(audio_fn,seg_fn)
                onoffs = np.loadtxt(seg_fn,usecols=(0,1))
            if len(onoffs.shape) == 1:
                onoffs = onoffs[None,:]
            
            a_ind = np.random.choice(len(onoffs))
            on,off = onoffs[a_ind,0],onoffs[a_ind,1]
            
            print(on,off)
            if '.wav' in audio_fn:
                sr,a = wavfile.read(audio_fn)
                
            elif '.flac' in audio_fn:
                a,sr = sf.read(audio_fn)

            
            orig_dtype = a.dtype
            on_ind,off_ind = int(round(on*sr)),int(round(off*sr))

            curr_len = off_ind - on_ind
            if curr_len < p['chunk length']:
                diff = p['chunk length'] - curr_len
                off_ind += diff 
                off += diff/sr
                print(f'extending segment by {diff/sr:.2f}s')
            orig_audio = a[on_ind:off_ind]
            if reduce_noise:
                noise_reduced_chunk_on = max(0,on_ind-sr)
                on_diff = on_ind - noise_reduced_chunk_on
                noise_reduced_chunk_off = min(len(a),off_ind+sr)
                off_diff = noise_reduced_chunk_off - off_ind
                a = nr.reduce_noise(y=a[noise_reduced_chunk_on:noise_reduced_chunk_off],sr=sr,prop_decrease=p['prop_reduce'],time_constant_s=0.4,stationary=False)
                orig_audio = a[on_diff:-off_diff]
            print(orig_audio.shape)
            print(len(orig_audio)/sr, off-on)

            t = np.arange(0,off-on,1/sr)[:len(orig_audio)]
            
            _,sx_orig,_,sx_freqs, *_ = ssq_stft(orig_audio,fs=sr)
            orig_spec = np.abs(sx_orig)
            vmin = np.amin(orig_spec)
            vmax = np.amax(orig_spec)
            vmax = (vmax - vmin)/10 + vmin
            cwt_kws = {'wavelet': (p['wavelet'],WAVELET_HP_DICT[p['wavelet']]),'nv':p['nv'],'scales':p['scales']}

            if p['squeeze_freqs']:
                band_min = p['band min']
                band_max = p['band max']
            else:
                band_min = 0
                band_max = int(round(sr)/2)
            if preprocess_type == 'ssq':
                                # processing ssq cwt
                recon_a,cwt_freqs,ssq_scaleogram = ssq_preprocess(orig_audio,t,cwt_kws,\
                                                        p['chunk length'],show=False,\
                                                            min_band=band_min,max_band=band_max)
            else:
                recon_a,cwt_freqs,ssq_scaleogram = band_pass_preprocess(orig_audio,p['chunk length'],low_cut=band_min,
                                               high_cut=band_max,fs=sr,return_full_ssq=True,
                                               kw=cwt_kws,tn=t,show=False,order=p['order'])
            recon_a = recon_a.astype(orig_dtype)
            wavfile.write('./test_wav.wav',rate=sr,data=recon_a)
            _,sx_recon,*_ = ssq_stft(recon_a,fs=sr)
            recon_spec = np.abs(sx_recon)
            _tuning_plot(orig_spec,recon_spec,t,sx_freqs,cwt_freqs,\
                        np.abs(ssq_scaleogram),p['band min'],p['band max'],\
                            vmin=vmin,vmax=vmax,save_loc=img_fn)
            
            resp = input('Continue? [y] or [s]top tuning or [r]etune params: ')
            if resp == 's':
                return p

def filter_by_tags(audio_files,seg_files,audio_tags,seg_tags):

    ### assumes tags, files are all sorted already

    #filtered_audio_files = []
    #filtered_seg_files = []

    all_endings = set(audio_tags).intersection(seg_tags)
    
    filtered_audio_files = [w for w,t in zip(audio_files,audio_tags) if t in all_endings]
    filtered_seg_files = [s for s,t in zip(seg_files,seg_tags) if t in all_endings]

    return filtered_audio_files,filtered_seg_files			

def preprocess_helper(in_dir,out_dir,hyperparameters,audio_ext,reprocess,preprocess_type,reduce_noise):

    print(f"processing directory {in_dir} \n")
    print(f"{'' if reprocess else 'not '}reprocessing files")
        
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    afs = glob.glob(os.path.join(in_dir,'*' + audio_ext))

    for af in afs:

        new_fn = af.split('/')[-1].split(audio_ext)[0] + '_cleaned.wav'
        new_fn = os.path.join(out_dir,new_fn)
        if os.path.isfile(new_fn) and not reprocess:
            continue

        if '.wav' in af:
            sr,orig_audio = wavfile.read(af)
        elif '.flac' in af:
            orig_audio,sr = sf.read(af)

        orig_dtype = orig_audio.dtype
        if reduce_noise:
            orig_audio = nr.reduce_noise(y=orig_audio,sr=sr,prop_decrease=hyperparameters['prop_reduce'],time_constant_s=0.4,stationary=False)

        cwt_kws = {'wavelet': (hyperparameters['wavelet'],WAVELET_HP_DICT[hyperparameters['wavelet']]),
                    'nv':hyperparameters['nv'],
                    'scales':hyperparameters['scales']}
        try:
            t = np.arange(0,len(orig_audio)/sr,1/sr)[:len(orig_audio)]
            if hyperparameters['squeeze_freqs']:
                if preprocess_type == 'ssq':
                    recon_a,*_ = ssq_preprocess(orig_audio,t,cwt_kws,\
                                                                hyperparameters['chunk length'],show=False,\
                                                                min_band=hyperparameters['band min'],\
                                                                max_band=hyperparameters['band max'],\
                                                                    return_full_ssq=False)
                elif preprocess_type == 'band-pass':
                    recon_a,*_ = band_pass_preprocess(orig_audio,hyperparameters['chunk length'],low_cut=hyperparameters['band min'],
                                                high_cut=hyperparameters['band max'],fs=sr,return_full_ssq=True,
                                                kw=cwt_kws,tn=t,show=False)
            else:
                recon_a = orig_audio

        except:
            print(f"error in processing {af}")
            print(t.shape)
            print(orig_audio.shape)
            assert False
        recon_a = recon_a.astype(orig_dtype)
        wavfile.write(new_fn,rate=sr,data=recon_a)

def preprocess(audio_dirs,out_dirs,hp_dict,audio_ext='.wav',parallel = False,reprocess=True,preprocess_type='ssq',reduce_noise=True):

    assert len(audio_dirs) == len(out_dirs), print(f"need one out dir per audio dir! {len(audio_dirs)} audio dirs and {len(out_dirs)} out dirs")
    #print(hp_dict)
    if parallel:
        n_jobs = int(os.cpu_count() // 2)
        gen = zip(audio_dirs,out_dirs,repeat(hp_dict),repeat(audio_ext),repeat(reprocess),repeat(preprocess_type),repeat(reduce_noise))
        Parallel(n_jobs = n_jobs)(delayed(preprocess_helper)(*args) for args in gen)
        
    else:
        for ii,(in_dir,out_dir) in enumerate(zip(audio_dirs,out_dirs)):

            preprocess_helper(in_dir,out_dir,hp_dict,audio_ext,reprocess,preprocess_type)

        














