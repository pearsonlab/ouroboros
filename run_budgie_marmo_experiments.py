import fire
from data.real_data import get_segmented_audio
from data.data_utils import get_loaders
from train.model_cv import model_cv_lambdas
from train.train import load_model
from train.eval import full_eval_model
import os
import numpy as np
import matplotlib.pyplot as plt
from visualization.model_vis import format_axes
import pickle
from analysis.analysis import get_budgie_fncs,get_marmo_fncs
import gc 
import glob

def run_experiments(budgie_data_path='',marmo_data_path='',
                    model_path='',seed=92):
    

    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    budgie_model_path = os.path.join(model_path,'budgies')
    marmo_model_path = os.path.join(model_path,'marmos')
    #### let's do budgie data first ##########

    print("budgie training and analysis")
    budgie_audio_path,budgie_seg_path= budgie_data_path,budgie_data_path

    audios,sr = get_segmented_audio(budgie_audio_path,budgie_seg_path,audio_subdir='',\
                                seg_subdir='',envelope=False,context_len=0.15,\
                                audio_type='_cleaned.wav',seg_type='izationTimestamp.txt',\
                                    max_pairs=3000,seed=seed)
    dls = get_loaders(np.vstack(audios),cv = True,train_size=0.6,seed=seed)

    budgie_model = model_cv_lambdas(dls,1/sr,\
                                    nEpochs=400,model_path=budgie_model_path)
    
    if os.path.isfile(os.path.join(budgie_model_path,'eval_data.pkl')):
        with open(os.path.join(budgie_model_path,'eval_data.pkl'),'rb') as f:
            budgie_eval_dict = pickle.load(f)
    else:
        budgie_r2s,budgie_best,budgie_resids,budgie_spec_ratio,budgie_specs,budgie_ext = full_eval_model(budgie_model,dls,audios,1/sr,use_results=False,\
                        n_int=50,plot_dir=budgie_model_path,plot_steps=False)
        
        budgie_eval_dict={'r2s':budgie_r2s,
                        'best_data':budgie_best,
                        'resids':budgie_resids,
                        'spec_ratio':budgie_spec_ratio,
                        'specs':budgie_specs,
                        'ext':budgie_ext}

        with open(os.path.join(budgie_model_path,'eval_data.pkl'),'wb') as f:
            pickle.dump(budgie_eval_dict,f)

    if os.path.isfile(os.path.join(budgie_model_path,'func_data.pkl')):
        with open(os.path.join(budgie_model_path,'func_data.pkl'),'rb') as f:
            budgie_func_dict = pickle.load(f)
    else:
        b_o,b_g,b_k,b_w,b_a = get_budgie_fncs(budgie_model,budgie_audio_path,budgie_seg_path,seed=seed,cut_percentile=75)

        budgie_func_dict = {
            'omegas':b_o,
            'gammas':b_g,
            'kernels':b_k,
            'weights':b_w,
            'audio':b_a
        }

        with open(os.path.join(budgie_model_path,'func_data,pkl'),'wb') as f:
            pickle.dump(budgie_func_dict,f)

    del audios
    del dls
    #del budgie_model
    #del budgie_eval_dict 
    #del budgie_func_dict() 
    gc.collect()


    #### then marmoset data
    print("Done!")
    print("Marmoset training and analysis")
    marmo_audio_path,marmo_seg_path = marmo_data_path,marmo_data_path
    marmo_audio_subdir='s6*/wavfiles/synchro_cleaned'
    marmo_seg_subdir = 's6*/wavfiles'

    audios,sr = get_segmented_audio(marmo_audio_path,marmo_seg_path,audio_subdir=marmo_audio_subdir,\
                                seg_subdir=marmo_seg_subdir,envelope=False,context_len=0.15,\
                                audio_type='_cleaned.wav',seg_type='.txt',\
                                    max_pairs=3000,seed=seed)
    
    dls = get_loaders(np.vstack(audios),cv = True,train_size=0.6,seed=seed)

    marmo_model = model_cv_lambdas(dls,1/sr,\
                                    nEpochs=200,model_path=marmo_model_path,
                                    n_layers=3,expand_factor=8)

    if os.path.isfile(os.path.join(marmo_model_path,'eval_data.pkl')):
        with open(os.path.join(marmo_model_path,'eval_data.pkl'),'rb') as f:
            marmo_eval_dict = pickle.load(f)
    else:
        marmo_r2s,marmo_best,marmo_resids,marmo_spec_ratio,marmo_specs,marmo_ext = full_eval_model(marmo_model,dls,audios,1/sr,use_results=False,\
                        n_int=50,plot_dir=marmo_model_path,plot_steps=False)
        
        marmo_eval_dict={'r2s':marmo_r2s,
                        'best_data':marmo_best,
                        'resids':marmo_resids,
                        'spec_ratio':marmo_spec_ratio,
                        'specs':marmo_specs,
                        'ext':marmo_ext}

        with open(os.path.join(marmo_model_path,'eval_data.pkl'),'wb') as f:
            pickle.dump(marmo_eval_dict,f)

    marmos = glob.glob(os.path.join(marmo_data_path,'s6*'))
    marmo_ids = [m.split('/')[-1] for m in marmos]

    marmo_func_dict = {}
    for id, path in zip(marmo_ids,marmos):

        
        wavs = glob.glob(os.path.join(path,'wavfiles','*.wav'))
        unique_vocs = set([w.split('_')[-1].split('.')[0] for w in wavs])
        marmo_func_dict[id] = {}

        for u in unique_vocs:
            print(f"getting vocalizations for marmo {id}: {u}")
            if os.path.isfile(os.path.join(marmo_model_path,f'{id}_{u}_func_data.pkl')):
                with open(os.path.join(marmo_model_path,f'{id}_{u}_func_data.pkl'),'rb') as f:
                    marmo_func_dict[id][u] = pickle.load(f)

            else:
                m_o,m_g,m_k,m_w,m_a = get_marmo_fncs(marmo_model,marmo_audio_path,\
                                                marmo_seg_path,marmo_name=id,\
                                                    voctype=u,seed=seed)
                
                marmo_func_dict[id][u] = {
                    'omegas':m_o,
                    'gammas':m_g,
                    'kernels':m_k,
                    'weights':m_w,
                    'audio':m_a
                }
                with open(os.path.join(marmo_model_path,f'{id}_{u}_func_data.pkl'),'wb') as f:
                    pickle.dump(marmo_func_dict[id][u],f)



    #### TO DO:: PLOT SOME STUFF RIGHT HERE
    
    f5_mosaic = \
    [['Marmo 2 sylls']*2 + ['Marmo 1 syll all ms']*2,\
     ['Marmo 2 sylls']*2 + ['Marmo 1 syll all ms']*2,\
     ['Budgie something']*4
    ]


if __name__ == '__main__':

    fire.Fire(run_experiments)


