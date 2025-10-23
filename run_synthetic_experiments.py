import fire
from data.real_data import get_segmented_audio
from data.toy_data import gen_stacks,load_coen_data
from data.data_utils import get_loaders
from train.model_cv import model_cv_lambdas
from train.train import load_model
from train.eval import full_eval_model
import os
import numpy as np
import matplotlib.pyplot as plt
from visualization.model_vis import format_axes
import pickle


def run_experiments(gabo_data_path='',coen_data_path='',\
                    adult_zf_model_path='',\
                        generate=False,seed=92,synth_model_path='',
                        save_freq=10):

    audio_path =os.path.join(gabo_data_path,'audio')
    seg_path =os.path.join(gabo_data_path,'segs')
    func_path =os.path.join(gabo_data_path,'funcs')

    gdp = os.path.join(synth_model_path,'gabo_results.pkl')
    cdp = os.path.join(synth_model_path,'coen_results.pkl')
    sdp = os.path.join(synth_model_path,'stack_results.pkl')

    assert os.path.isdir(adult_zf_model_path),print(adult_zf_model_path, "not found!!")
    if generate:
        from data.generate_data_gabo import generate_vocal_dataset
        import jax.random as jr
        key = jr.key(seed=seed)
        _ = generate_vocal_dataset(key,n_vocs=1000,\
                                   audio_loc=audio_path,
                                   seg_loc=seg_path,
                                   func_loc=func_path,)

    if not os.path.isdir(synth_model_path):
        os.mkdir(synth_model_path)

    if not os.path.isfile(gdp):
    
        print("Training model on Gabo's synthetic vocalizations")
        audios,sr = get_segmented_audio(audio_path,seg_path,audio_subdir='',\
                                    seg_subdir='',envelope=False,context_len=0.1,\
                                    audio_type='.wav',seg_type='.txt',max_pairs=3000,seed=seed)

        dls = get_loaders(np.vstack(audios),cv = True,train_size=0.6,seed=seed)
        gabo_path = os.path.join(synth_model_path,'gabo_model')
        gabo_model = model_cv_lambdas(dls,1/sr,\
                            nEpochs=100,model_path=gabo_path,
                            save_freq=save_freq)

        gabo_r2s,gabo_best,gabo_resids,gabo_spec_ratio,gabo_specs,gabo_ext = full_eval_model(gabo_model,dls,audios,1/sr,use_results=False,\
                        n_int=50,plot_dir=gabo_path,plot_steps=False)
        
        gabo_data_dict={'r2s':gabo_r2s,
                        'best_data':gabo_best,
                        'resids':gabo_resids,
                        'spec_ratio':gabo_spec_ratio,
                        'specs':gabo_specs,
                        'ext':gabo_ext}
        
        with open(gdp,'wb') as f:
            pickle.dump(gabo_data_dict,f)
    
    else:

        print("loading Gabo's results")

        with open(gdp,'rb') as f:
            gabo_data_dict = pickle.load(f)

            gabo_r2s = gabo_data_dict['r2s']
            gabo_best = gabo_data_dict['best_data']
            gabo_resids = gabo_data_dict['resids']
            gabo_spec_ratio = gabo_data_dict['spec_ratio']
            gabo_specs = gabo_data_dict['specs']
            gabo_ext = gabo_data_dict['ext']


    #assert False

    if not os.path.isfile(sdp):
        print("Training model on synthetic stacks")

        sr_stack=44100
        stacks,d_stack,d2_stack = gen_stacks(n_samples=2000,sample_rate=sr_stack)
        dls = get_loaders(np.vstack(stacks)[:,:,None],cv = True,train_size=0.6,seed=seed)
        stack_path = os.path.join(synth_model_path,'stackies')
        stack_model = model_cv_lambdas(dls,1/sr_stack,\
                                nEpochs=100,model_path=stack_path,
                                save_freq=save_freq,tau=1/10000)
        stack_r2s,stack_best,stack_resids,stack_spec_ratio,stack_specs,stack_ext = full_eval_model(stack_model,dls,stacks,1/sr_stack,use_results=False,\
                    n_int=50,plot_dir=stack_path,plot_steps=False)
        
        stack_data_dict={'r2s':stack_r2s,
                        'best_data':stack_best,
                        'resids':stack_resids,
                        'spec_ratio':stack_spec_ratio,
                        'specs':stack_specs,
                        'ext':stack_ext}
        with open(sdp,'wb') as f:
            pickle.dump(stack_data_dict,f)
        
    else:
        print("loading stack results")
        with open(sdp,'rb') as f:
            stack_data_dict = pickle.load(f)

            stack_r2s = stack_data_dict['r2s']
            stack_best = stack_data_dict['best_data']
            stack_resids = stack_data_dict['resids']
            stack_spec_ratio = stack_data_dict['spec_ratio']
            stack_specs = stack_data_dict['specs']
            stack_ext = stack_data_dict['ext']

        
    if not os.path.isfile(cdp):
        print("Assessing model on Coen's synthetic syrinx")

        coen_data = load_coen_data(coen_data_path,target_fs=44100)
        #print(coen_data[4])
        adult_model,*_=load_model(adult_zf_model_path,kernel_type='full_poly') # here, let's use use cv_better_weighed_adultzf_92 0.05
        coen_r2s,coen_best,coen_resids,coen_spec_ratio,coen_specs,coen_ext = full_eval_model(adult_model,None,coen_data[1],1/coen_data[4],use_results=False,\
                        n_int=50,plot_dir=synth_model_path,plot_steps=False)
        coen_data_dict={'r2s':coen_r2s,
                        'best_data':coen_best,
                        'resids':coen_resids,
                        'spec_ratio':coen_spec_ratio,
                        'specs':coen_specs,
                        'ext':coen_ext}
        with open(cdp,'wb') as f:
            pickle.dump(coen_data_dict,f)
        
    else:
        print("Loading Coen's results")
        with open(cdp,'rb') as f:
            coen_data_dict = pickle.load(f)

            coen_r2s = coen_data_dict['r2s']
            coen_best = coen_data_dict['best_data']
            coen_resids = coen_data_dict['resids']
            coen_spec_ratio = coen_data_dict['spec_ratio']
            coen_specs = coen_data_dict['specs']
            coen_ext = coen_data_dict['ext']
    
    
    full_mosaic = \
    [['r2_stack','d2_stack_real','d2_stack_real','d2_stack_pred','d2_stack_pred','d2_stack_resid','d2_stack_resid','stack_resid_hist','stack_resid_hist'],
     ['pix_ratio_stack','spec_stack_real','spec_stack_real','spec_stack_emp','spec_stack_emp','spec_stack_md2','spec_stack_md2','spec_stack_m','spec_stack_m'],
     ['r2_g','d2_g_real','d2_g_real','d2_g_pred','d2_g_pred','d2_g_resid','d2_g_resid','g_resid_hist','g_resid_hist'],
     ['pix_ratio_g','spec_g_real','spec_g_real','spec_g_emp','spec_g_emp','spec_g_md2','spec_g_md2','spec_g_m','spec_g_m'],
     ['r2_c','d2_c_real','d2_c_real','d2_c_pred','d2_c_pred','d2_c_resid','d2_c_resid','c_resid_hist','c_resid_hist'],
     ['pix_ratio_c','spec_c_real','spec_c_real','spec_c_emp','spec_c_emp','spec_c_md2','spec_c_md2','spec_c_m','spec_c_m']]
    fig_full = plt.figure(layout='constrained',figsize=(30,30))
    full_ax_dict = fig_full.subplot_mosaic(full_mosaic)
    
    ####### stack plots ##########
    ############# 2nd derivs ##########
    full_ax_dict['r2_stack'].hist(stack_r2s,bins=100,density=True)
    full_ax_dict['d2_stack_real'].plot(stack_best[0][0],stack_best[0][1],color='tab:blue')
    ylim_real_s = full_ax_dict['d2_stack_real'].get_ylim()
    full_ax_dict['d2_stack_pred'].plot(stack_best[1][0],stack_best[1][1],color='tab:orange')
    full_ax_dict['d2_stack_pred'].set_ylim(ylim_real_s)
    full_ax_dict['d2_stack_resid'].plot(stack_resids[0],stack_resids[1])
    full_ax_dict['d2_stack_resid'].set_ylim(ylim_real_s)
    full_ax_dict['stack_resid_hist'].hist(stack_resids[1],bins=100,density=True)

    ############# specs ###############
    full_ax_dict['pix_ratio_stack'].hist(stack_spec_ratio[0],bins=100,density=True)
    full_ax_dict['pix_ratio_stack'].hist(stack_spec_ratio[1],bins=100,density=True)
    full_ax_dict['spec_stack_real'].imshow(stack_specs[0],origin='lower',aspect='auto',extent=stack_ext)
    full_ax_dict['spec_stack_emp'].imshow(stack_specs[1],origin='lower',aspect='auto',extent=stack_ext)
    full_ax_dict['spec_stack_md2'].imshow(stack_specs[2],origin='lower',aspect='auto',extent=stack_ext)
    full_ax_dict['spec_stack_m'].imshow(stack_specs[3],origin='lower',aspect='auto',extent=stack_ext)
    #sd = np.nanstd(stack_resids[1])
    #px = lambda x: (1/np.sqrt(2*np.pi*sd**2))*np.exp(-x**2/(2*sd**2))
    #xlims = full_ax_dict['stack_resid_hist'].get_xlim()
    #xax = np.linspace(xlims[0],xlims[1],1000)
    #yax = px(xax)
    #full_ax_dict['stack_resid_hist'].plot(xax,yax,color='tab:red')


    ######### TO DO: ADD INSET TO D2 PLOTS ###########
    ####### gabo plots ###########
    ############# 2nd derivs ##########
    full_ax_dict['r2_g'].hist(gabo_r2s,bins=100,density=True)
    full_ax_dict['d2_g_real'].plot(gabo_best[0][0],gabo_best[0][1],color='tab:blue')
    ylim_real_g = full_ax_dict['d2_g_real'].get_ylim()
    full_ax_dict['d2_g_pred'].plot(gabo_best[1][0],gabo_best[1][1],color='tab:orange')
    full_ax_dict['d2_g_pred'].set_ylim(ylim_real_g)
    full_ax_dict['d2_g_resid'].plot(gabo_resids[0],gabo_resids[1])
    full_ax_dict['d2_g_resid'].set_ylim(ylim_real_g)
    full_ax_dict['g_resid_hist'].hist(gabo_resids[1],bins=100,density=True)

    ############# specs ###############
    full_ax_dict['pix_ratio_g'].hist(gabo_spec_ratio[0],bins=100,density=True)
    full_ax_dict['pix_ratio_g'].hist(gabo_spec_ratio[1],bins=100,density=True)
    full_ax_dict['spec_g_real'].imshow(gabo_specs[0],origin='lower',aspect='auto',extent=gabo_ext)
    full_ax_dict['spec_g_emp'].imshow(gabo_specs[1],origin='lower',aspect='auto',extent=gabo_ext)
    full_ax_dict['spec_g_md2'].imshow(gabo_specs[2],origin='lower',aspect='auto',extent=gabo_ext)
    full_ax_dict['spec_g_m'].imshow(gabo_specs[3],origin='lower',aspect='auto',extent=gabo_ext)


    ####### coen plots ###########
    ############# 2nd derivs ##########
    full_ax_dict['r2_stack'].hist(coen_r2s,bins=100,density=True)
    full_ax_dict['d2_c_real'].plot(coen_best[0][0],coen_best[0][1],color='tab:blue')
    ylim_real_c = full_ax_dict['d2_c_real'].get_ylim()
    full_ax_dict['d2_c_pred'].plot(coen_best[1][0],coen_best[1][1],color='tab:orange')
    full_ax_dict['d2_c_pred'].set_ylim(ylim_real_c)
    full_ax_dict['d2_c_resid'].plot(coen_resids[0],coen_resids[1])
    full_ax_dict['d2_c_resid'].set_ylim(ylim_real_c)
    full_ax_dict['c_resid_hist'].hist(coen_resids[1],bins=100,density=True)

    ############# specs ###############
    full_ax_dict['pix_ratio_c'].hist(coen_spec_ratio[0],bins=100,density=True)
    full_ax_dict['pix_ratio_c'].hist(coen_spec_ratio[1],bins=100,density=True)
    full_ax_dict['spec_c_real'].imshow(coen_specs[0],origin='lower',aspect='auto',extent=coen_ext)
    full_ax_dict['spec_c_emp'].imshow(coen_specs[1],origin='lower',aspect='auto',extent=coen_ext)
    full_ax_dict['spec_c_md2'].imshow(coen_specs[2],origin='lower',aspect='auto',extent=coen_ext)
    full_ax_dict['spec_c_m'].imshow(coen_specs[3],origin='lower',aspect='auto',extent=coen_ext)

    for key in full_ax_dict.keys():
        format_axes(full_ax_dict[key])
        if 'spec' in key:
            full_ax_dict[key].set_xticks([])
            full_ax_dict[key].set_yticks([])
    #plt.tight_layout()
    plt.savefig(os.path.join(synth_model_path,'big_full_plot.svg'))
    plt.close()


if __name__ == '__main__':


    fire.Fire(run_experiments)