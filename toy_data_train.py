from data.toy_data import *
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
import os

def run_model(model_path= '',\
                n_samples=4000,sr=44100,alpha=8.,\
                    theta=2.,sample_rate=44100,noise_sd=0.025,\
                nEpochs=100, n_kernels=10,seed=None,\
                    smooth_len=0.005,vis_freq=100,tau=1000,
                    lr=1e-3,oversample_prop = 1,smoothing=False,\
                        n_layers=2,expand_factor=4,d_conv=4,d_state=1,lam=1.5,mult_factor_kernels=4):

    d,d1,d2 = gen_stacks(n_samples,alpha=alpha,\
                         theta=theta,sample_rate=sample_rate,\
                            noise_sd=noise_sd)

    d = d[:,:,None]
    print(f"splitting {len(d)} samples into dataloaders")

    print(f'getting dataloaders with seed {seed}')
    dls = get_loaders(d,cv = True,train_size=0.6,\
                      seed=seed,oversample_prop=oversample_prop,\
                        dt=1/sample_rate)

    print("training comparison models end to end: poly first")    

    kernel = fullPolyModule(nTerms=n_kernels,device='cuda',x_dim=1,z_dim=2,activation = lambda x: x,lam=lam,trend_filtering=True)
    reg_weights=True
    
    full_model_poly=rkhs_ouroboros(d_data=1,n_layers=n_layers,d_state=d_state,\
                d_conv=d_conv,expand_factor=expand_factor,tau=tau,\
                            smooth_len=smooth_len,kernel=kernel)

    full_opt_poly = Adam(full_model_poly.parameters(),
                lr=lr)
    full_scheduler_poly = ReduceLROnPlateau(full_opt_poly,factor=0.5,patience=max(nEpochs//25,2),min_lr=1e-10)
    model_path_full_poly = model_path + f'/kernelboros_poly_toy_end_to_end'
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
    kernel = simpleGaussModule(nTerms=mult_factor_kernels*n_kernels,device='cuda',x_dim=1,z_dim=2,activation=lambda x: x,trend_filtering=True)
    reg_weights=False

    full_model_gauss=rkhs_ouroboros(d_data=1,n_layers=n_layers,d_state=d_state,\
                d_conv=d_conv,expand_factor=expand_factor,tau=tau,\
                            smooth_len=smooth_len,kernel=kernel)

    full_opt_gauss = Adam(full_model_gauss.parameters(),
                lr=lr)
    full_scheduler_gauss = ReduceLROnPlateau(full_opt_gauss,factor=0.5,patience=max(nEpochs//25,2),min_lr=1e-10)
    model_path_full_gauss = model_path + f'/kernelborous_gauss_toy_end_to_end'
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