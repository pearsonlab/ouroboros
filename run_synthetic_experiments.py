import fire
from data.real_data import get_segmented_audio
from data.toy_data import gen_stacks,load_coen_data
from data.data_utils import get_loaders
from train.model_cv import model_cv_lambdas
from train.train import load_model
import os
import numpy as np


def run_experiments(gabo_data_path='',coen_data_path='',\
                    adult_zf_model_path='',\
                        generate=False,seed=92):

    audio_path =os.path.join(gabo_data_path,'audio')
    seg_path =os.path.join(gabo_data_path,'segs')
    func_path =os.path.join(gabo_data_path,'funcs')

    if generate:
        from data.generate_data_gabo import generate_vocal_dataset
        import jax.random as jr
        key = jr.key(seed=seed)
        _ = generate_vocal_dataset(key,n_vocs=1000,\
                                   audio_loc=audio_path,
                                   seg_loc=seg_path,
                                   func_loc=func_path)

   
    audios,sr = get_segmented_audio(audio_path,seg_path,audio_subdir='',\
                                seg_subdir='',envelope=False,context_len=0.1,\
                                audio_type='.wav',seg_type='.txt',max_pairs=3000,seed=seed)

    dls = get_loaders(np.vstack(audios),cv = True,train_size=0.6,seed=seed)
    gabo_model = model_cv_lambdas(dls,1/sr)

    full_eval_model(gabo_model,dls,audios,1/sr)

    sr_stack=44100
    stacks,d_stack,d2_stack = gen_stacks(n_samples=2000,sample_rate=sr_stack)
    dls = get_loaders(np.vstack(audios),cv = True,train_size=0.6,seed=seed)
    stack_model = model_cv_lambdas(dls,1/sr)
    full_eval_model(stack_model,dls,stacks,1/sr_stack)

    coen_data = load_coen_data(coen_data_path)
    adult_model,*_=load_model(adult_zf_model_path)
    full_eval_model(adult_model,None,coen_data[1],coen_data[4])

if __name__ == '__main__':


    fire.Fire(run_experiments)