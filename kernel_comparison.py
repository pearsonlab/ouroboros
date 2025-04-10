from data.real_data import *
from data.data_utils import get_loaders
from train.train import train,save_model,load_model
from model.constrained_model import rkhs_ouroboros,simple_ouroboros
from model.kernels import *
import matplotlib.pyplot as plt
import gc
import torch
import fire 
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from visualization.model_vis import loss_plot,r2_plot
from train.eval import eval_model_error,assess_kernels
from utils import sse

def run_model(audio_path,seg_path='', model_path= '',\
              seg_filetype='.txt',audio_filetype='.wav',voctype='adultsong',\
                context_len=0.3,max_pairs=1000,trend_level=1,
                nEpochs=100, kernel_type='gauss',n_kernels=10,alpha=1e7,seed=None,\
                    save_loaders=False,smooth_len=0.005,vis_freq=100,tau=1000,
                    lr=1e-3,oversample_prop = 1,smoothing=False,only_full=False,\
                        n_layers=2,expand_factor=4,d_conv=4,d_state=1,lam=1.5,mult_factor_kernels=4):

    
    use_trend = True if trend_level > 0 else False
    if seg_path == '':
        seg_path = audio_path

    model_path += '_' + str(seed)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
 
    print(f'loading data from {audio_path}, {seg_path}')
    audios,sr = get_segmented_audio(audio_path,seg_path,envelope=False,context_len=context_len,\
                                    audio_type=audio_filetype,seg_type=seg_filetype,max_pairs=max_pairs,seed=seed)

    print(f"splitting {len(audios)} samples into dataloaders")
    loader_path = model_path + '/loaders.pth'
    if os.path.isfile(loader_path):
        dls = torch.load(loader_path,weights_only=False)
    else:
        print(f'getting dataloaders with seed {seed}')
        dls = get_loaders(np.vstack(audios),cv = True,train_size=0.6,seed=seed,oversample_prop=oversample_prop,dt=1/sr)
        if save_loaders:
            print('saving dataloaders...')
            del audios
            dls['sr'] = sr
            torch.save(dls,loader_path)

    #alpha_xaxis = np.arange(len(alphas))
    #n_kernels = [10] #1,2,3

    ## if data dim = 1 and stacking on data dim, z_dim should be 4
    ## if data dim = 1 and stacking on time dim, z_dim should be 2
    

    print("training comparison models end to end: poly first")    

    kernel = fullPolyModule(nTerms=n_kernels,device='cuda',x_dim=1,z_dim=2,activation = lambda x: x,lam=lam,trend_filtering=use_trend)
    reg_weights=True
    
    full_model_poly=rkhs_ouroboros(d_data=1,n_layers=n_layers,d_state=d_state,\
                d_conv=d_conv,expand_factor=expand_factor,tau=tau,\
                            smooth_len=smooth_len,kernel=kernel)

    full_opt_poly = Adam(full_model_poly.parameters(),
                lr=lr)
    full_scheduler_poly = ReduceLROnPlateau(full_opt_poly,factor=0.5,patience=max(nEpochs//25,2),min_lr=1e-10)
    model_path_full_poly = model_path + f'/kernelborous_poly_{voctype}_end_to_end'
    save_loc_poly = model_path_full_poly + '/checkpoint_100.tar'
    if os.path.isfile(save_loc_poly):
        #print(f'loading_checkpoint at {save_loc}')
        full_model_poly,full_opt_poly,full_scheduler_poly = load_model(save_loc_poly,kernel_type='full_poly')

    else:


        tl,vl,full_model_poly,full_opt_poly = train(full_model_poly,full_opt_poly,loss_fn=lambda y,yhat: sse(yhat,y,reduction='mean'),\
                                loaders=dls,scheduler=full_scheduler_poly,nEpochs=nEpochs,val_freq=1,\
                        runDir=model_path_full_poly,\
                        dt = 1/sr,vis_freq=vis_freq,smoothing=smoothing,\
                            reg_weights=reg_weights)
        
        loss_plot(tl,vl,save_loc=model_path_full_poly,show=False)

        
        save_model(full_model_poly,full_opt_poly,save_loc_poly,n_layers=n_layers,d_state=d_state,expand_factor=expand_factor,d_conv=d_conv)
    print('evaluating end-to-end model poly....')
    (train_mu_poly,test_mu_poly),(train_sd_poly,test_sd_poly) = eval_model_error(dls,full_model_poly,dt=1/sr,comparison='test')
    assess_kernels(dls['train'],full_model_poly,dt=1/sr,saveDir=model_path_full_poly)
    print("training comparison models end to end: then rbf kernel")
    kernel = simpleGaussModule(nTerms=mult_factor_kernels*n_kernels,device='cuda',x_dim=1,z_dim=2,activation=lambda x: x,trend_filtering=use_trend)
    reg_weights=False

    full_model_gauss=rkhs_ouroboros(d_data=1,n_layers=n_layers,d_state=d_state,\
                d_conv=d_conv,expand_factor=expand_factor,tau=tau,\
                            smooth_len=smooth_len,kernel=kernel)

    full_opt_gauss = Adam(full_model_gauss.parameters(),
                lr=lr)
    full_scheduler_gauss = ReduceLROnPlateau(full_opt_gauss,factor=0.5,patience=max(nEpochs//25,2),min_lr=1e-10)
    model_path_full_gauss = model_path + f'/kernelborous_gauss_{voctype}_end_to_end'
    save_loc_gauss = model_path_full_gauss + '/checkpoint_100.tar'
    if os.path.isfile(save_loc_gauss):
        #print(f'loading_checkpoint at {save_loc}')
        full_model_gauss,full_opt_gauss,full_scheduler_gauss = load_model(save_loc_gauss,kernel_type='gauss')

    else:


        tl,vl,full_model_gauss,full_opt_gauss = train(full_model_gauss,full_opt_gauss,loss_fn=lambda y,yhat: sse(yhat,y,reduction='mean'),\
                                loaders=dls,scheduler=full_scheduler_gauss,nEpochs=nEpochs,val_freq=1,\
                        runDir=model_path_full_gauss,\
                        dt = 1/sr,vis_freq=vis_freq,smoothing=smoothing,\
                            reg_weights=reg_weights)
        
        loss_plot(tl,vl,save_loc=model_path_full_gauss,show=False)

        
        save_model(full_model_gauss,full_opt_gauss,save_loc_gauss,n_layers=n_layers,d_state=d_state,expand_factor=expand_factor,d_conv=d_conv)

    print('evaluating end-to-end model rbf....')
    (train_mu_rbf,test_mu_rbf),(train_sd_rbf,test_sd_rbf) = eval_model_error(dls,full_model_gauss,dt=1/sr,comparison='test')
    assess_kernels(dls['train'],full_model_poly,dt=1/sr,saveDir=model_path_full_gauss)

    #(train_coef,test_coef),(train_coef_sd,test_coef_sd) = eval_model_integration(dls,full_model,dt=1/sr,n_segs=25,st=0)

    r2_plot([train_mu_poly,test_mu_poly,train_mu_rbf,test_mu_rbf],\
            [train_sd_poly,test_sd_poly,train_sd_rbf,test_sd_rbf],\
            labels=['Train (poly)', 'Test (poly)', 'Train (RBF)', 'Test (RBF)'],\
                saveloc=model_path,show=False)

    return

if __name__ == '__main__':

    fire.Fire(run_model)
    print('all done!')