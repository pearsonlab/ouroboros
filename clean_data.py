from data.preprocess import tune_preprocessing, preprocess,filter_by_tags,HP_DICT
import time
import glob
import os
from fire import Fire
#import noisereduce as nr


def preprocess_data(audio_loc,seg_loc,out_ext,\
                    audio_subdirs='',seg_subdirs='',\
                        audio_ext='.wav',seg_ext='.txt',parallel = True,
                        reprocess=True,
                        clean_type='ssq',
                        reduce_noise=True):
    

    ## default: grabs all subdirs

    audio_dirs = glob.glob(os.path.join(audio_loc,audio_subdirs))
    seg_dirs = glob.glob(os.path.join(seg_loc,seg_subdirs))
    audio_dirs.sort()
    seg_dirs.sort()
    #out_dirs = [os.path.join(o,out_ext) for o in audio_dirs]

    ## this part assumes that the top subdirectory in audio_subdirs is the one matched across 
    ## audio, seg
    split_aud_sub = audio_subdirs.split('/')
    split_seg_sub = seg_subdirs.split('/')
    #print(split_aud_sub,split_seg_sub)

    aud_sub_depth=len(split_aud_sub)
    seg_sub_depth=len(split_seg_sub)

    audio_tags = [a.split('/')[-aud_sub_depth] for a in audio_dirs]
    seg_tags = [s.split('/')[-seg_sub_depth] for s in seg_dirs]

    #print(audio_tags,seg_tags)
    audio_dirs,seg_dirs  = filter_by_tags(audio_dirs,seg_dirs,audio_tags,seg_tags)
    if len(audio_dirs) == 0:
        print("Lost all directories!")
        print(f"Tried to get dirs with audio tags: {audio_tags}")
        print(f"Tried to get dirs with seg tags: {seg_tags}")
        assert False
    #print(len(audio_dirs),len(seg_dirs))
    #print(audio_dirs,seg_dirs)
    out_dirs = [os.path.join(o,out_ext) for o in audio_dirs]
    #print(audio_dirs[:5],seg_dirs[:5])
    #print(audio_ext,seg_ext)
    audio_files = sum([glob.glob(os.path.join(a,'*' + audio_ext)) for a in audio_dirs],[])
    seg_files = sum([glob.glob(os.path.join(s,'*' + seg_ext)) for s in seg_dirs],[])
    audio_files.sort()
    seg_files.sort()
    audio_tags = [a.split(audio_ext)[0].split('/')[-1] for a in audio_files]
    seg_tags = [s.split(seg_ext)[0].split('/')[-1] for s in seg_files]

    audio_files,seg_files = filter_by_tags(audio_files,seg_files,audio_tags,seg_tags)
    if len(audio_files) == 0:
        print("Lost all files!")
        print(f"Tried to get files with audio tags: {audio_tags}")
        print(f"Tried to get files with seg tags: {seg_tags}")
        assert False
    #valid = True
    #for tag in seg_tags:
    #    try:
    #        assert tag in audio_tags
    #    except:
    #        print(f"{tag} not in audio tags")
    #        valid = False
    #for tag in audio_tags:
    #    try:
    #        assert tag in seg_tags
    #    except:
    #        print(f"{tag} not in seg tags")
    #        valid = False 

    true_segs = []
    
    #sassert valid
    #print(audio_files,seg_files)
    #print(audio_files[:5],seg_files[:5])
    assert len(audio_files) > 0, print("no audio files! check your directories & subdirs")
    #print(len(audio_files),len(seg_files))
    hps = tune_preprocessing(audio_files,seg_files,HP_DICT,preprocess_type=clean_type,reduce_noise=reduce_noise)

    print('now cleaning data!')
    start = time.time()
    preprocess(audio_dirs,out_dirs,hps,audio_ext=audio_ext,
               parallel=parallel,reprocess=reprocess,
               preprocess_type=clean_type,reduce_noise=reduce_noise)
    end = time.time()
    print(f"preprocessed your data in {end - start :.2f}s! If you have other files to preprocess, it'll probably take that long")

if __name__ == '__main__':

    Fire(preprocess_data)