from data.preprocess import tune_preprocessing, preprocess, HP_DICT
import time
import glob
import os
from fire import Fire
import yaml


def preprocess_data(
    audio_dir:str,
    seg_dir:str,
    out_dir:str,
    audio_id:str='.wav',
    parallel:bool=True,
    reprocess:bool=True,
    clean_type:str="ssq",
    reduce_noise:bool=True,
    ):

    audio_files = glob.glob(os.path.join(audio_dir,'*'+audio_id))
    assert len(audio_files) > 0, print(
        "no audio files! check your directories & subdirs"
    )
    audio_tags = [a.split('/')[-1].split(audio_id)[0] for a in audio_files]
    seg_files = [os.path.join(seg_dir,a) for a in audio_tags]

    # tunes parameters for preprocessing
    hps = tune_preprocessing(
        audio_files,
        seg_files,
        HP_DICT,
        preprocess_type=clean_type,
        reduce_noise=reduce_noise,
    )

    print("now cleaning data!")
    start = time.time()
    preprocess(
        audio_files,
        out_dir,
        hps,
        parallel=parallel,
        reprocess=reprocess,
        preprocess_type=clean_type,
        reduce_noise=reduce_noise,
    )
    end = time.time()
    print(
        f"preprocessed your data in {end - start:.2f}s! If you have other files to preprocess, it'll probably take that long"
    )

if __name__ == "__main__":
    Fire(preprocess_data)