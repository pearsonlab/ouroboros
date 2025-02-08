from data.real_data import *
from data.data_utils import get_loaders
from train.train import train,eval_model
from constrained_model import rkhs_ouroboros
from kernels import *
import matplotlib.pyplot as plt
import gc
import torch
import fire 
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle


def run_model(audio_path,seg_path='', model_path= '',\
              seg_filetype='.txt',audio_filetype='.wav',voctype='adultsong',\
                context_len=0.3,max_pairs=1000,trend_level=1,
                nEpochs=100, kernel_type='gauss'):
    

    use_trend = True if trend_level > 0 else False
    if seg_path == '':
        seg_path = audio_path
 
    print('loading data')
    audios,sr = get_segmented_audio(audio_path,seg_path,envelope=False,context_len=context_len,\
                                    audio_type=audio_filetype,seg_type=seg_filetype,max_pairs=max_pairs)
    
    dls = get_loaders(np.vstack(audios),cv = False,train_size=0.6)

    alphas = [0,1,2,4,8,16,32,64,128]
    n_kernels = [1,2,3,4,5,10,20]
    train_cv_list,test_cv_list = [],[]
    for ii,alpha in enumerate(alphas):
        train_cv_list.append([])
        test_cv_list.append([])

        for n_kernel in n_kernels:
            if kernel_type == 'gauss':
                kernel = simpleGaussModule(nTerms=n_kernel,device='cuda',xdim=1,z_dim=4,activation=lambda x: x)
            else:
                kernel = polyModule(nTerms=n_kernel,device='cuda',x_dim=1,z_dim=4,activation = lambda x: x,lam=0.9)
            
            model = rkhs_ouroboros(d_data=1,n_layers=1,d_state=1,\
                        d_conv=4,expand_factor=1,tau=1000,\
                                    smooth_len=0.005,kernel=kernel)
            opt = Adam(model.parameters(),
                        lr=1e-3)
            scheduler = ReduceLROnPlateau(opt,factor=0.75,patience=5,min_lr=1e-10)

            model_path_full = model_path + f'/kernelborous_gauss10_{voctype}_smoothedweights_fittod2y_scaled_trendfiltering_{trend_level}_alpha_{alpha}_nkernels_{n_kernel}'

            tl,vl,model,opt = train(model,opt,loss_fn=torch.nn.MSELoss(),loaders=dls,scheduler=scheduler,nEpochs=nEpochs,val_freq=1,\
                            runDir=model_path_full,\
                            dt = 1/sr,use_trend_filtering=True,trend_level=trend_level,vis_freq=0,\
                                alpha=alpha)
            
            (train_mu,test_mu),(train_sd,test_sd) = eval_model(dls,model,dt=1/sr)
            train_cv_list[ii].append((train_mu,train_sd))
            test_cv_list[ii].append((test_mu,test_sd))

            sd = {'ouroboros':model.state_dict(),
            'opt':opt.state_dict()}
            save_loc = model_path_full + '/checkpoint_100.tar'
            torch.save(sd,save_loc)

    with open('/home/miles/Downloads/train_cv_vals.pkl','wb') as pfile:
        pickle.dump(train_cv_list,pfile)

    with open('/home/miles/Downloads/test_cv_vals.pkl','wb') as pfile:
        pickle.dump(test_cv_list,pfile)


    return


if __name__ == '__main__':

    fire.Fire(run_model)