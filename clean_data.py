from data.preprocess import tune_preprocessing, preprocess,HP_DICT

import glob
import os
from fire import Fire


def preprocess_data(audio_loc,seg_loc,out_ext,\
                    audio_subdirs='',seg_subdirs='',\
                        audio_ext='.wav',seg_ext='.txt'):
    

    ## default: grabs all subdirs



    audio_dirs = glob.glob(os.path.join(audio_loc,audio_subdirs))
    seg_dirs = glob.glob(os.path.join(seg_loc,seg_subdirs))
    out_dirs = [os.path.join(o,out_ext) for o in audio_dirs]

    print(audio_dirs[:5],seg_dirs[:5])
    audio_files = sum([glob.glob(os.path.join(a,'*' + audio_ext)) for a in audio_dirs],[])
    seg_files = sum([glob.glob(os.path.join(s,'*' + seg_ext)) for s in seg_dirs],[])
    #print(audio_files,seg_files)
    print(audio_files[:5],seg_files[:5])
    hps = tune_preprocessing(audio_files,seg_files,HP_DICT)

    preprocess(audio_dirs,out_dirs,hps,audio_ext=audio_ext)

if __name__ == '__main__':

    Fire(preprocess_data)