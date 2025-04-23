from data.preprocess import tune_preprocessing, preprocess,HP_DICT

import glob
import os
from fire import Fire


def preprocess_data(audio_loc,seg_loc,out_ext,\
                    audio_subdirs='',seg_subdirs='',\
                        audio_ext='.wav',seg_ext='.txt'):
    




    audio_files = sum([glob.glob(os.path.join(a,'*' + audio_ext)) for a in audio_dirs],[])
    seg_files = sum([glob.glob(os.path.join(s,'*' + seg_ext)) for s in seg_dirs],[])
    print(audio_files,seg_files)
    hps = tune_preprocessing(audio_files,seg_files,HP_DICT)

if __name__ == '__main__':

    Fire(preprocess_data)