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
from visualization.model_vis import loss_plot
from train.eval import eval_model_error,eval_model_integration

def run_model(audio_path,seg_path='', model_path= '',\
              seg_filetype='.txt',audio_filetype='.wav',voctype='adultsong',\
                context_len=0.3,max_pairs=1000,trend_level=1,
                nEpochs=100, kernel_type='gauss',n_kernels=10,alpha=1e7,n_models=10,\
                    save_loaders=False,smooth_len=0.005):

    
    use_trend = True if trend_level > 0 else False
    if seg_path == '':
        seg_path = audio_path

    #model_path += '_' + str(seed)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
 
    print('loading data')
    
    audios,sr = get_segmented_audio(audio_path,seg_path,envelope=False,context_len=context_len,\
                                    audio_type=audio_filetype,seg_type=seg_filetype,max_pairs=max_pairs)
    
    kernel_train_cv_err = []
    kernel_test_cv_err = []

    kernel_train_cv_sd = []
    kernel_test_cv_sd = []

    kernel_train_int_coef = []
    kernel_test_int_coef = []

    kernel_train_int_coef_sd = []
    kernel_test_int_coef_sd = []
    model_axis=np.arange(1,n_models + 1)
    for m in range(n_models):
        print(f'getting dataloaders for model {m+1}')
        dls = get_loaders(np.vstack(audios),cv = True,train_size=0.6)
       

        if kernel_type == 'gauss':
            kernel = simpleGaussModule(nTerms=n_kernels,device='cuda',x_dim=1,z_dim=4,activation=lambda x: x,trend_filtering=use_trend)
        else:
            kernel = polyModule(nTerms=n_kernels,device='cuda',x_dim=1,z_dim=4,activation = lambda x: x,lam=0.9,trend_filtering=use_trend)
        
        model = rkhs_ouroboros(d_data=1,n_layers=1,d_state=1,\
                    d_conv=4,expand_factor=1,tau=1000,\
                                smooth_len=smooth_len,kernel=kernel,
                                trend_filtering=use_trend)
        opt = Adam(model.parameters(),
                    lr=1e-3)
        scheduler = ReduceLROnPlateau(opt,factor=0.75,patience=5,min_lr=1e-10)

        model_path_full = model_path + f'/kernelborous_{voctype}_trendfiltering_{trend_level}_alpha_{alpha}_kernel_{kernel_type}_nkernels_{n_kernels}_model_{m}'
        save_loc = model_path_full + '/checkpoint_100.tar'

        if os.path.isfile(save_loc):
            #print(f'loading_checkpoint at {save_loc}')
            state = torch.load(save_loc,weights_only=True)
            model.load_state_dict(state['ouroboros'])
            opt.load_state_dict(state['opt'])
        else:


            tl,vl,model,opt = train(model,opt,loss_fn=torch.nn.MSELoss(),loaders=dls,scheduler=scheduler,nEpochs=nEpochs,val_freq=1,\
                            runDir=model_path_full,\
                            dt = 1/sr,use_trend_filtering=use_trend,trend_level=trend_level,vis_freq=0,\
                                alpha=alpha)
            
            loss_plot(tl,vl,save_loc=model_path_full,show=False)
            sd = {'ouroboros':model.state_dict(),
            'opt':opt.state_dict()}
            
            torch.save(sd,save_loc)
        #model.trend_filtering=False
        (train_mu,test_mu),(train_sd,test_sd) = eval_model_error(dls,model,dt=1/sr)
        kernel_train_cv_err.append(train_mu)
        kernel_test_cv_err.append(test_mu)

        kernel_train_cv_sd.append(train_sd)
        kernel_test_cv_sd.append(test_sd)

        (train_coef,test_coef),(train_coef_sd,test_coef_sd) = eval_model_integration(dls,model,dt=1/sr,n_segs=25,st=0)
        kernel_train_int_coef.append(train_coef)
        kernel_test_int_coef.append(test_coef)
        kernel_train_int_coef_sd.append(train_coef_sd)
        kernel_test_int_coef_sd.append(test_coef_sd)
    
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

    ax = plt.gca()
    ax.bar(model_axis,kernel_train_cv_err,0.25,color='tab:blue',label='train')
    ax.bar(model_axis+0.5,kernel_test_cv_err,0.25,color='tab:orange',label='validation')
    ax.errorbar(model_axis,kernel_train_cv_err,yerr=kernel_train_cv_sd,capsize=4,fmt='o',color='k')
    ax.errorbar(model_axis+0.5,kernel_test_cv_err,yerr=kernel_test_cv_sd,capsize=4,fmt='o',color='k')
    ax.set_xticks(model_axis+0.25,[f"model {n}" for n in model_axis])
    ax.set_xlabel("Model number")
    ax.set_ylabel(r"$R^2$")
    ax.legend()
    plt.savefig(os.path.join(model_path,f'train_test_repro_error_kernel_{kernel_type}_nkernels_{n_kernels}.svg'))
    plt.close()

    ax = plt.gca()
    ax.bar(model_axis,kernel_train_int_coef,0.25,color='tab:blue',label='train')
    ax.bar(model_axis+0.5,kernel_test_int_coef,0.25,color='tab:orange',label='validation')
    ax.errorbar(model_axis,kernel_train_int_coef,yerr=kernel_train_int_coef_sd,capsize=4,fmt='o',color='k')
    ax.errorbar(model_axis+0.5,kernel_test_int_coef,yerr=kernel_test_int_coef_sd,capsize=4,fmt='o',color='k')
    ax.set_xticks(model_axis+0.25,[f"model {n}" for n in model_axis])
    ax.set_xlabel("Model number")
    ax.set_ylabel("Integration error coefficient value")
    ax.legend()
    plt.savefig(os.path.join(model_path,f'train_test_repro_int_coef_kernel_{kernel_type}_nkernels_{n_kernels}.svg'))
    plt.close()


    ####### add in train/test error analysis here #######
    ####### maybe save out dataloader so that we can 
    ####### maintain same train/test split after training run -- that's a good idea actually, save in model path

    #return model, dls

if __name__ == '__main__':

    fire.Fire(run_model)
    print('all done!')