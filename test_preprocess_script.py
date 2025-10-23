from data.preprocess import tune_preprocessing,HP_DICT,preprocess 
import glob
import os


def test_preprocess(audio_dirs,seg_dirs,audio_ext='.wav',seg_ext='.txt'):

    #print(audio_dirs,seg_dirs)
    #assert os.path.isdir
    audio_files = sum([glob.glob(os.path.join(a,'*' + audio_ext)) for a in audio_dirs],[])
    seg_files = sum([glob.glob(os.path.join(s,'*' + seg_ext)) for s in seg_dirs],[])
    print(audio_files,seg_files)
    hps = tune_preprocessing(audio_files,seg_files,HP_DICT)
    return hps

if __name__ == '__main__':

    ads = ['/home/miles/isilon/All_Staff/birds/long/Budgie']
    sds = ['/home/miles/isilon/All_Staff/birds/long/Budgie']
    hps = test_preprocess(audio_dirs=ads,\
                    seg_dirs=sds,\
                        audio_ext='.flac')
    out_dirs = [os.path.join(a,'cleaned_audio') for a in ads]
    
    preprocess(ads,sds,hps,audio_ext='.flac')


