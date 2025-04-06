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
from visualization.model_vis import loss_plot
from train.eval import eval_model_error,eval_model_integration
from utils import sse

def eval_model(model_path= '',\
         context_len=0.3,max_pairs=1000,
         kernel_type='gauss',
         n_kernels=10,n_layers=2,
         expand_factor=4,d_state=1,
         tau=1000,smooth_len=0.005):
    
    use_trend=True
    if kernel_type == 'gauss':
        kernel = simpleGaussModule(nTerms=n_kernels,device='cuda',x_dim=1,z_dim=2,activation=lambda x: x,trend_filtering=use_trend)
    elif kernel_type == 'constant_gauss':
        kernel = constantGaussModule(nTerms=n_kernels,device='cuda',x_dim=1,z_dim=2,activation=lambda x: x,trend_filtering=use_trend)
    else:
        kernel = polyModule(nTerms=n_kernels,device='cuda',x_dim=1,z_dim=2,activation = lambda x: x,lam=lam,trend_filtering=use_trend)
    
    full_model=rkhs_ouroboros(d_data=1,n_layers=n_layers,d_state=d_state,\
                d_conv=4,expand_factor=expand_factor,tau=tau,\
                            smooth_len=smooth_len,kernel=kernel)

    birds = ['blk521','org512','org545','pur567']

    path = '/home/miles/isilon/All_Staff/birds/mooney/CAGbirds'

    for bird in birds:
        p=os.path.join(path,bird,'data')
        audios,sr = get_segmented_audio(p,p,envelope=False,
                                        context_len=context_len,
                                        audio_type='2*[0-9][0-9]/denoised/*.wav',
                                        seg_type='2*[0-9][0-9]/denoised_segments/*.txt',
                                        max_pairs=max_pairs)
        dls = get_loaders(np.vstack(audios),cv = True,train_size=0.6,seed=None,oversample_prop=1,dt=1/sr)


