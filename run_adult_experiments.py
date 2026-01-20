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
from analysis.analysis import get_zf_fncs
import gc 
import glob

def run_experiments(adult_d_ud_path='',
                    adult_p_t_path='',model_path='',seed=92,
                    tau=1e-4,lr=1e-3,n_kernels=10,
                    max_segs=3000,nEpochs=200,
                    dir_undir=True,
                    pupil_tutor=True,batch_size=32):
    

    ## for the adult syll path, we're going to use blu285 or something idk
    ## maybe one of ofer's birds instead

    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    

    dir_undir_modelpath = os.path.join(model_path,'dir_undir')
    pupil_tutor_modelpath = os.path.join(model_path,'pupil_tutor')
    if not os.path.isdir(dir_undir_modelpath):
        os.mkdir(dir_undir_modelpath)
    if not os.path.isdir(pupil_tutor_modelpath):
        os.mkdir(pupil_tutor_modelpath)
        
    if dir_undir:
        ##### Directed, undirected models:
        dud_aud = os.path.join(adult_d_ud_path,'original_data')
        dud_seg = os.path.join(adult_d_ud_path,'song_segs')

    
        audios,sr = get_segmented_audio(dud_aud,dud_seg,audio_subdir='[U,D]*/double_denoised',\
                            seg_subdir='[U,D]*',envelope=False,context_len=0.15,\
                            audio_type='_cleaned.wav',seg_type='.txt',\
                                max_pairs=max_segs,seed=seed)
        #print(sr)
        dls = get_loaders(np.vstack(audios),cv = True,train_size=0.6,seed=seed,batch_size=batch_size)

        #print(np.amax(audios),np.amin(audios))
        dud_model = model_cv_lambdas(dls,1/sr,\
                                        nEpochs=nEpochs,model_path=dir_undir_modelpath,
                                        n_layers=3,expand_factor=8,n_kernels=n_kernels,
                                        tau=1/sr,lr=lr)
        
        dud_r2s,dud_best,dud_resids,dud_spec_ratio,dud_specs,dud_ext = full_eval_model(dud_model,dls,audios,1/sr,use_results=False,\
                        n_int=50,plot_dir=dir_undir_modelpath,plot_steps=False)
        
        dud_eval_dict={'r2s':dud_r2s,
                        'best_data':dud_best,
                        'resids':dud_resids,
                        'spec_ratio':dud_spec_ratio,
                        'specs':dud_specs,
                        'ext':dud_ext}

        with open(os.path.join(dir_undir_modelpath,'eval_data.pkl'),'wb') as f:
            pickle.dump(dud_eval_dict,f)

        """
        TO-DO: UPDATE THESE FUNCTIONS TO REFLECT CHANGES TO MODEL

        d_o,d_g,d_k,d_w = get_zf_fncs(dud_model,dud_aud,dud_seg,\
                                    aud_subdir='D*/synchro_cleaned_v1',\
                                        seg_subdir='D*',seed=seed)
        ud_o,ud_g,ud_k,ud_w = get_zf_fncs(dud_model,dud_aud,dud_seg,\
                                    aud_subdir='D*/synchro_cleaned_v1',\
                                        seg_subdir='D*',seed=seed)
        dir_func_dict = {
                    'omegas':d_o,
                    'gammas':d_g,
                    'kernels':d_k,
                    'weights':d_w
                }
        with open(os.path.join(dir_undir_modelpath,'dir_func_data.pkl'),'wb') as f:
            pickle.dump(dir_func_dict,f)

        undir_func_dict = {
                    'omegas':ud_o,
                    'gammas':ud_g,
                    'kernels':ud_k,
                    'weights':ud_w
                }
        with open(os.path.join(dir_undir_modelpath,'undir_func_data.pkl'),'wb') as f:
            pickle.dump(undir_func_dict,f)

        """
        del audios
        del dls 
        gc.collect()
    ############### pupil/tutor data ###############
    if pupil_tutor:
        pt_aud,pt_seg = adult_p_t_path,adult_p_t_path
        #birds =glob.glob(os.path.join(adult_p_t_path,'[b,g,p]*[0-9]'))
        birds = ['blk411','blk415','blk417','blk424','blk430','blk435','grn395','pur436']
        n_per_bird = max_segs//(2*len(birds))

        ####### need to get segments for this!!
        # found da segments
        ## label da segments, add some functionality for your segment label sylls to allow 
        # labeling prelabelled syllables
        audios=[]
        prev_sr = -1
        for b in birds:
            b = os.path.join(adult_p_t_path,b)
            tutor_aud,tutor_seg = b +'_tutor', b+'_tutor'
            audio_subdir = 'motif_audio_tutor/double_denoised'
            seg_subdir = 'motif_txt_tutor'
            print(f"Now loading {n_per_bird} from {tutor_aud.split('/')[-1]}")
            a,sr = get_segmented_audio(tutor_aud,tutor_seg,\
                                audio_subdir=audio_subdir,\
                                seg_subdir=seg_subdir,envelope=False,context_len=0.15,\
                                audio_type='_cleaned.wav',seg_type='.txt',\
                                    max_pairs=n_per_bird,seed=seed)
            if prev_sr == -1:
                prev_sr = sr
            assert sr == prev_sr
            audios.append(np.vstack(a))
            pupil_aud,pupil_seg = b,b
            audio_subdir='motif_audio/double_denoised'
            seg_subdir='motif_txt'
            print(f"Now loading {n_per_bird} from {pupil_aud.split('/')[-1]}")

            a,sr = get_segmented_audio(pupil_aud,pupil_seg,\
                                audio_subdir=audio_subdir,\
                                seg_subdir=seg_subdir,envelope=False,context_len=0.15,\
                                audio_type='_cleaned.wav',seg_type='.txt',\
                                    max_pairs=n_per_bird,seed=seed)
            assert sr == prev_sr
            audios.append(np.vstack(a))
        dls = get_loaders(np.vstack(audios),cv = True,train_size=0.6,seed=seed,batch_size=batch_size)

        pt_model = model_cv_lambdas(dls,1/sr,\
                                        nEpochs=200,model_path=pupil_tutor_modelpath,
                                        n_layers=3,expand_factor=8,n_kernels=n_kernels,
                                        tau=1/sr,lr=lr)
        
        pt_r2s,pt_best,pt_resids,pt_spec_ratio,pt_specs,pt_ext = full_eval_model(pt_model,dls,audios,1/sr,use_results=False,\
                        n_int=50,plot_dir=dir_undir_modelpath,plot_steps=False)
        
        pt_eval_dict={'r2s':pt_r2s,
                        'best_data':pt_best,
                        'resids':pt_resids,
                        'spec_ratio':pt_spec_ratio,
                        'specs':pt_specs,
                        'ext':pt_ext}
        
        with open(os.path.join(pupil_tutor_modelpath,'eval_data.pkl'),'wb') as f:
            pickle.dump(pt_eval_dict,f)
    
    #birds = glob.glob(os.path.join(adult_p_t_path,'*[0-9]'))
    """
    pupil_dicts = {}
    tutor_dicts= {}
    for bird in birds:
        b = bird.split('/')[-1]
        tutor = b + '_tutor'


        p_o,p_g,p_k,p_w = get_zf_fncs(pt_model,pt_aud,pt_seg,\
                                  aud_subdir=f'{b}/motif_audio/synchro_cleaned',\
                                    seg_subdir=f'{b}/motif_segs',seed=seed)
        t_o,t_g,t_k,t_w = get_zf_fncs(pt_model,pt_aud,pt_seg,\
                                  aud_subdir=f'{tutor}/motif_audio_tutor/synchro_cleaned',\
                                    seg_subdir=f'{tutor}/motif_segs',seed=seed)

        pupil_func_dict = {
                'omegas':p_o,
                'gammas':p_g,
                'kernels':p_k,
                'weights':p_w
            }
        pupil_dicts[b] = pupil_func_dict
        with open(os.path.join(pupil_tutor_modelpath,'pupil_func_data.pkl'),'wb') as f:
            pickle.dump(pupil_func_dict,f)

        tutor_func_dict = {
                    'omegas':t_o,
                    'gammas':t_g,
                    'kernels':t_k,
                    'weights':t_w
                }
        tutor_dicts[b] = tutor_func_dict
        with open(os.path.join(pupil_tutor_modelpath,'tutor_func_data.pkl'),'wb') as f:
            pickle.dump(tutor_func_dict,f) 

        """


if __name__ == '__main__':

    fire.Fire(run_experiments)