from data.real_data import *
from data.data_utils import get_loaders
from train.train import train,save_model,load_model
from model.constrained_model import rkhs_ouroboros,simple_ouroboros
from model.kernels import *
import matplotlib.pyplot as plt
import gc
import torch
import fire 
import os
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from visualization.model_vis import loss_plot
from train.eval import eval_model_error,eval_model_integration
from utils import sse

def run_model(audio_path,seg_path='', model_path= '',\
              seg_filetype='.txt',audio_filetype='.wav',voctype='adultsong',\
                context_len=0.3,max_pairs=1000,trend_level=1,
                nEpochs=100, kernel_type='gauss',n_kernels=10,alpha=1e7,seed=None,\
                    save_loaders=False,smooth_len=0.005,vis_freq=100,tau=1000,
                    lr=1e-3,oversample_prop = 1,smoothing=False,only_full=False):

    
    use_trend = True if trend_level > 0 else False
    if seg_path == '':
        seg_path = audio_path

    model_path += '_' + str(seed)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
 
    print('loading data')
    
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
    if not only_full:
        model = simple_ouroboros(d_data=1,n_layers=1,d_state=1,\
                    d_conv=4,expand_factor=1,tau=tau,\
                                smooth_len=smooth_len)
        opt = Adam(model.parameters(),
                    lr=lr)
        scheduler = ReduceLROnPlateau(opt,factor=0.75,patience=5,min_lr=1e-10)

        model_path_simple = model_path + f'/simpleborous_{voctype}'
        save_loc = model_path_simple + '/checkpoint_100.tar'

        if os.path.isfile(save_loc):
            #print(f'loading_checkpoint at {save_loc}')
            model,opt,scheduler = load_model(save_loc,kernel_type=kernel_type)

        else:


            tl,vl,model,opt = train(model,opt,loss_fn=lambda y,yhat: sse(yhat,y,reduction='mean'),loaders=dls,scheduler=scheduler,nEpochs=nEpochs,val_freq=1,\
                            runDir=model_path_simple,\
                            dt = 1/sr,vis_freq=vis_freq,smoothing=smoothing)
            
            loss_plot(tl,vl,save_loc=model_path_simple,show=False)
            
            save_model(model,opt,save_loc)

        
        print('performance after training omega, gamma')
        (train_mu,test_mu),(train_sd,test_sd) = eval_model_error(dls,model,dt=1/sr)

        #(train_coef,test_coef),(train_coef_sd,test_coef_sd) = eval_model_integration(dls,model,dt=1/sr,n_segs=25,st=0)

        if smoothing:
            use_trend=False

        if kernel_type == 'gauss':
            kernel = simpleGaussModule(nTerms=n_kernels,device='cuda',x_dim=1,z_dim=2,activation=lambda x: x,trend_filtering=use_trend)
        elif kernel_type == 'constant_gauss':
            kernel = constantGaussModule(nTerms=n_kernels,device='cuda',x_dim=1,z_dim=2,activation=lambda x: x,trend_filtering=use_trend)
        else:
            kernel = polyModule(nTerms=n_kernels,device='cuda',x_dim=1,z_dim=2,activation = lambda x: x,lam=0.9,trend_filtering=use_trend)
        
        full_model=rkhs_ouroboros(d_data=1,n_layers=1,d_state=1,\
                    d_conv=4,expand_factor=1,tau=tau,\
                                smooth_len=smooth_len,kernel=kernel)
        print('loading pre-trained kernelboros')
        full_model.load_omega_gamma(save_loc)

        full_opt = Adam(full_model.parameters(),
                    lr=lr)
        full_scheduler = ReduceLROnPlateau(full_opt,factor=0.75,patience=5,min_lr=1e-10)
        model_path_full = model_path_simple + f'/simpleborous_{voctype}_with_kernel'
        save_loc = model_path_full + '/checkpoint_100.tar'
        if os.path.isfile(save_loc):
            #print(f'loading_checkpoint at {save_loc}')
            full_model,full_opt,full_scheduler = load_model(save_loc,kernel_type=kernel_type)

        else:


            tl,vl,model,opt = train(full_model,full_opt,loss_fn=lambda y,yhat: sse(yhat,y,reduction='mean'),loaders=dls,scheduler=full_scheduler,nEpochs=nEpochs,val_freq=1,\
                            runDir=model_path_full,\
                            dt = 1/sr,vis_freq=vis_freq,smoothing=smoothing)
            
            loss_plot(tl,vl,save_loc=model_path_full,show=False)

            
            save_model(full_model,full_opt,save_loc)
        

        ####### add in train/test error analysis here #######
        ####### maybe save out dataloader so that we can 
        ####### maintain same train/test split after training run -- that's a good idea actually, save in model path
        print('evaluating full model....')
        (train_mu,test_mu),(train_sd,test_sd) = eval_model_error(dls,full_model,dt=1/sr)

        #(train_coef,test_coef),(train_coef_sd,test_coef_sd) = eval_model_integration(dls,full_model,dt=1/sr,n_segs=25,st=0)
    else:
        model_path_simple = model_path

    print("training comparison model end to end")

    if kernel_type == 'gauss':
        kernel = simpleGaussModule(nTerms=n_kernels,device='cuda',x_dim=1,z_dim=2,activation=lambda x: x,trend_filtering=use_trend)
        reg_weights=False
    elif kernel_type == 'constant_gauss':
        kernel = constantGaussModule(nTerms=n_kernels,device='cuda',x_dim=1,z_dim=2,activation=lambda x: x,trend_filtering=use_trend)
        reg_weights=False
    elif kernel_type == 'full_poly':
        kernel = fullPolyModule(nTerms=n_kernels,device='cuda',x_dim=1,z_dim=2,activation = lambda x: x,lam=10,trend_filtering=use_trend)
        reg_weights=True
        print(kernel.activation)
    else:
        kernel = polyModule(nTerms=n_kernels,device='cuda',x_dim=1,z_dim=2,activation = lambda x: x,lam=0.9,trend_filtering=use_trend)
        reg_weights=False
    full_model=rkhs_ouroboros(d_data=1,n_layers=2,d_state=1,\
                d_conv=4,expand_factor=4,tau=tau,\
                            smooth_len=smooth_len,kernel=kernel)

    full_opt = Adam(full_model.parameters(),
                lr=lr)
    full_scheduler = ReduceLROnPlateau(full_opt,factor=0.5,patience=5,min_lr=1e-10)
    model_path_full = model_path_simple + f'/kernelborous_{voctype}_end_to_end'
    save_loc = model_path_full + '/checkpoint_100.tar'
    if os.path.isfile(save_loc):
        #print(f'loading_checkpoint at {save_loc}')
        full_model,full_opt,full_scheduler = load_model(save_loc,kernel_type=kernel_type)

    else:


        tl,vl,model,opt = train(full_model,full_opt,loss_fn=lambda y,yhat: sse(yhat,y,reduction='mean'),loaders=dls,scheduler=full_scheduler,nEpochs=nEpochs,val_freq=1,\
                        runDir=model_path_full,\
                        dt = 1/sr,vis_freq=vis_freq,smoothing=smoothing,\
                            reg_weights=reg_weights)
        
        loss_plot(tl,vl,save_loc=model_path_full,show=False)

        
        save_model(full_model,full_opt,save_loc)
    print('evaluating end-to-end model....')
    (train_mu,test_mu),(train_sd,test_sd) = eval_model_error(dls,full_model,dt=1/sr)

    #(train_coef,test_coef),(train_coef_sd,test_coef_sd) = eval_model_integration(dls,full_model,dt=1/sr,n_segs=25,st=0)

    return model, dls

if __name__ == '__main__':

    fire.Fire(run_model)
    print('all done!')