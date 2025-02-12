from data.real_data import *
from data.data_utils import get_loaders
from train.train import train
from train.eval import eval_model_error,eval_model_integration
from constrained_model import rkhs_ouroboros
from kernels import *
import matplotlib.pyplot as plt
import gc
import torch
import fire 
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle


def run_model(audio_path,seg_path='', model_path= '',plot_path='',\
              seg_filetype='.txt',audio_filetype='.wav',voctype='adultsong',\
                context_len=0.3,max_pairs=1000,trend_level=1,
                nEpochs=100, kernel_type='gauss'):
    

    use_trend = True if trend_level > 0 else False
    if seg_path == '':
        seg_path = audio_path
 
    print('loading data')
    audios,sr = get_segmented_audio(audio_path,seg_path,envelope=False,context_len=context_len,\
                                    audio_type=audio_filetype,seg_type=seg_filetype,max_pairs=max_pairs)
    
    dls = get_loaders(np.vstack(audios),cv = True,train_size=0.6)

    alphas = [0,1,2,4,8,16,32,64,128]
    alpha_xaxis = np.arange(len(alphas))
    n_kernels = [4,5,10,20] #1,2,3
    train_cv_list,test_cv_list = [],[]
    for ii,n_kernel in enumerate(n_kernels):
        kernel_train_cv_err = []
        kernel_test_cv_err = []

        kernel_train_cv_sd = []
        kernel_test_cv_sd = []

        kernel_train_int_coef = []
        kernel_test_int_coef = []
        for alpha in tqdm(alphas,desc='Iterating through alphas',total=len(alphas)):
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
            save_loc = model_path_full + '/checkpoint_100.tar'

            if os.path.isfile(save_loc):
                #print(f'loading_checkpoint at {save_loc}')
                state = torch.load(save_loc,weights_only=True)
                model.load_state_dict(state['ouroboros'])
                opt.load_state_dict(state['opt'])
            else:


                tl,vl,model,opt = train(model,opt,loss_fn=torch.nn.MSELoss(),loaders=dls,scheduler=scheduler,nEpochs=nEpochs,val_freq=1,\
                                runDir=model_path_full,\
                                dt = 1/sr,use_trend_filtering=True,trend_level=trend_level,vis_freq=0,\
                                    alpha=alpha)
                
                sd = {'ouroboros':model.state_dict(),
                'opt':opt.state_dict()}
                
                torch.save(sd,save_loc)
            
            (train_mu,test_mu),(train_sd,test_sd) = eval_model_error(dls,model,dt=1/sr)
            kernel_train_cv_err.append(train_mu)
            kernel_test_cv_err.append(test_mu)

            kernel_train_cv_sd.append(train_sd)
            kernel_test_cv_sd.append(test_sd)

            train_coef,test_coef = eval_model_integration(dls,model,dt=1/sr,n_segs=25,st=0)
            kernel_train_int_coef.append(train_coef)
            kernel_test_int_coef.append(test_coef)

        min_err_ind = np.argmin(kernel_test_cv_err)
        min_int_ind = np.argmin(kernel_test_int_coef)
        print(f"best error alpha for {n_kernel} kernels: {alphas[min_err_ind]}")
        print(f"best integration alpha for {n_kernel} kernels: {alphas[min_int_ind]}")
        ax = plt.gca()
        ax.bar(alpha_xaxis,kernel_train_cv_err,0.25,color='tab:blue',label='train')
        ax.bar(alpha_xaxis+0.5,kernel_test_cv_err,0.25,color='tab:orange',label='validation')
        ax.errorbar(alpha_xaxis,kernel_train_cv_err,yerr=kernel_train_cv_sd,capsize=4,fmt='o',color='k')
        ax.errorbar(alpha_xaxis+0.5,kernel_test_cv_err,yerr=kernel_test_cv_sd,capsize=4,fmt='o',color='k')
        ax.set_xticks(alpha_xaxis+0.25,alphas)
        ax.set_xlabel("Trend filtering weight")
        ax.set_ylabel("MSE")
        ax.legend()
        plt.savefig(os.path.join(plot_path,f'train_test_error_nkernels_{n_kernel}.svg'))
        plt.close()

            

    with open('/home/miles/Downloads/train_cv_vals.pkl','wb') as pfile:
        pickle.dump(train_cv_list,pfile)

    with open('/home/miles/Downloads/test_cv_vals.pkl','wb') as pfile:
        pickle.dump(test_cv_list,pfile)


    return


if __name__ == '__main__':

    fire.Fire(run_model)