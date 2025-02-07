from data.real_data import *
from data.data_utils import get_loaders
from train.train import train
from constrained_model import rkhs_ouroboros
from kernels import *
import matplotlib.pyplot as plt
import gc
import torch
import fire 
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

def run_model(audio_path,seg_path='', model_path= '',\
              seg_filetype='.txt',audio_filetype='.wav',voctype='adultsong',\
                context_len=0.3,max_pairs=1000,trend_level=1,
                nEpochs=100, kernel_type='gauss',n_kernels=10):

    
    use_trend = True if trend_level > 0 else False
    if seg_path == '':
        seg_path = audio_path
 
    print('loading data')
    audios,sr = get_segmented_audio(audio_path,seg_path,envelope=False,context_len=context_len,\
                                    audio_type=audio_filetype,seg_type=seg_filetype,max_pairs=max_pairs)
    
    dls = get_loaders(np.vstack(audios))


    if kernel_type == 'gauss':
        kernel = simpleGaussModule(nTerms=n_kernels,device='cuda',xdim=1,z_dim=4,activation=lambda x: x)
    else:
        kernel = polyModule(nTerms=n_kernels,device='cuda',x_dim=1,z_dim=4,activation = lambda x: x,lam=0.9)
    
    model = rkhs_ouroboros(d_data=1,n_layers=1,d_state=1,\
                  d_conv=4,expand_factor=1,tau=1000,\
                             smooth_len=0.005,kernel=kernel)
    opt = Adam(model.parameters(),
                lr=1e-3)
    scheduler = ReduceLROnPlateau(opt,factor=0.75,patience=5,min_lr=1e-10)

    model_path_full = model_path + f'/kernelborous_gauss10_{voctype}_smoothedweights_fittod2y_scaled_trendfiltering_{trend_level}'

    tl,vl,model,opt = train(model,opt,loss_fn=torch.nn.MSELoss(),loaders=dls,scheduler=scheduler,nEpochs=nEpochs,val_freq=1,\
                       runDir=model_path_full,\
                       dt = 1/sr,use_trend_filtering=use_trend,trend_level=trend_level,vis_freq=0)
    
    sd = {'ouroboros':model.state_dict(),
      'opt':opt.state_dict()}
    save_loc = model_path_full + '/checkpoint_100.tar'
    torch.save(sd,save_loc)

    
    tl = np.array(tl)
    ax = plt.gca()
    ax.plot(tl[:,0],color='tab:blue',label='train loss')

    vl = np.array(vl)
    ax.plot(vl[:,0],vl[:,1],color='tab:orange',label='val loss')
    ax.set_title("losses")
    plt.legend()
    ax.set_xlabel("gradient updates")
    ax.set_ylabel("loss (MSE)")
    plt.savefig(model_path_full + '/train_losses.svg')
    plt.close()


    ####### add in train/test error analysis here #######
    ####### maybe save out dataloader so that we can 
    ####### maintain same train/test split after training run -- that's a good idea actually, save in model path

    loader_path = model_path_full + '/loaders.pth'
    torch.save(dls,loader_path)
    return model, dls

if __name__ == '__main__':

    fire.Fire(run_model)