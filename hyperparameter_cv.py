from data.real_data import *
from data.data_utils import get_loaders
from train.train import train,load_model,save_model
from train.eval import eval_model_error
from train.model_cv import model_cv_lambdas
from model.constrained_model import rkhs_ouroboros
from model.kernels import *
import matplotlib.pyplot as plt
import pandas as pd
import fire 
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
from visualization.model_vis import loss_plot
from utils import sse


def run_model(audio_path,seg_path='', model_path= '',\
              audio_subdir='',seg_subdir='',\
              seg_filetype='.txt',audio_filetype='.wav',voctype='adultsong',\
                context_len=0.3,max_pairs=1000,
                nEpochs=100, seed=None,smooth_len=0.005,vis_freq=100,tau=1000,
                    lr=1e-3,smoothing=False,\
                        n_kernels=30,n_layers=2,expand_factor=4,d_conv=4,d_state=1):

    

    use_trend = 1
    if seg_path == '':
        seg_path = audio_path

    model_path += '_' + str(seed)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
 
    print('loading data')
    
    audios,sr = get_segmented_audio(audio_path,seg_path,audio_subdir=audio_subdir,\
                                    seg_subdir=seg_subdir,envelope=False,context_len=context_len,\
                                    audio_type=audio_filetype,seg_type=seg_filetype,max_pairs=max_pairs,seed=seed)
    
   
    print(f'getting dataloaders with seed {seed}')
    dls = get_loaders(np.vstack(audios),cv = True,train_size=0.6,seed=seed)

    model_cv_lambdas(dls,1/sr,nEpochs=nEpochs,lr=lr,\
                    n_kernels=n_kernels,expand_factor=expand_factor,\
                    n_layers=n_layers,d_state=d_state,d_conv=d_conv,\
                    tau=tau,smooth_len=smooth_len,\
                    model_path=model_path)
    """
    ## model cv lambda here
    lambdas = np.array([0.01,0.05,0.1,0.25,0.5,0.75,1.]) #* 10**2.5
    #im going to ASSUME that with lambda = 1, regularization is 
    # on the order of 10^13, loss on the order of 10^18 -- so multiply all these by 10^2.5 (since we end up multiplying by lambda^2)
    lambda_xaxis = np.arange(len(lambdas))
    #n_kernels = 30 #1,2,3
    lam_train_cv_err = []
    lam_test_cv_err = []

    lam_train_cv_sd = []
    lam_test_cv_sd = []
    for ii,lam in enumerate(lambdas):
        
        print(f"Regularizing with lambda={lam}")

        kernel = fullPolyModule(nTerms=n_kernels,device='cuda',x_dim=1,z_dim=2,activation = lambda x: x,lam=lam,trend_filtering=use_trend)
        reg_weights=True    
        full_model_poly=rkhs_ouroboros(d_data=1,n_layers=n_layers,d_state=d_state,\
                d_conv=d_conv,expand_factor=expand_factor,tau=tau,\
                            smooth_len=smooth_len,kernel=kernel)

        full_opt_poly = Adam(full_model_poly.parameters(),
                    lr=lr)
        full_scheduler_poly = ReduceLROnPlateau(full_opt_poly,factor=0.5,patience=max(nEpochs//25,2),min_lr=1e-10)
        model_path_full_poly = model_path + f'/kernelborous_poly_{voctype}_end_to_end_lambda_{lam}'
        save_loc_poly = model_path_full_poly + '/checkpoint_100.tar'
        

        if os.path.isfile(save_loc_poly):
            full_model_poly,full_opt_poly,full_scheduler_poly = load_model(save_loc_poly,kernel_type='full_poly')

        else:


            tl,vl,full_model_poly,full_opt_poly = train(full_model_poly,full_opt_poly,loss_fn=lambda y,yhat: sse(yhat,y,reduction='mean'),\
                                loaders=dls,scheduler=full_scheduler_poly,nEpochs=nEpochs,val_freq=1,\
                        runDir=model_path_full_poly,\
                        dt = 1/sr,vis_freq=vis_freq,smoothing=smoothing,\
                            reg_weights=reg_weights)
        
            loss_plot(tl,vl,save_loc=model_path_full_poly,show=False)

            
            save_model(full_model_poly,full_opt_poly,save_loc_poly,n_layers=n_layers,d_state=d_state,expand_factor=expand_factor,d_conv=d_conv)
        
        
        (train_mu,test_mu),(train_sd,test_sd) = eval_model_error(dls,full_model_poly,dt=1/sr)
        lam_train_cv_err.append(train_mu)
        lam_test_cv_err.append(test_mu)

        lam_train_cv_sd.append(train_sd)
        lam_test_cv_sd.append(test_sd)

    min_err_ind = np.argmax(lam_test_cv_err) # argmax, since 'err' is actually r2
    print(f"best R2 alpha for {n_kernels} kernels: {lambdas[min_err_ind]}")
    ax = plt.gca()
    ax.bar(lambda_xaxis,lam_train_cv_err,0.25,color='tab:blue',label='train')
    ax.bar(lambda_xaxis+0.5,lam_test_cv_err,0.25,color='tab:orange',label='validation')
    ax.errorbar(lambda_xaxis,lam_train_cv_err,yerr=lam_train_cv_sd,capsize=4,fmt='o',color='k')
    ax.errorbar(lambda_xaxis+0.5,lam_test_cv_err,yerr=lam_test_cv_sd,capsize=4,fmt='o',color='k')
    ax.set_xticks(lambda_xaxis+0.25,lambdas)
    ax.set_xlabel("Polynomial degree penalty")
    ax.set_ylabel(r"$R^2$")
    ax.legend()
    plt.savefig(os.path.join(model_path,f'train_test_error_kernel_poly_nkernels_30.svg'))
    plt.close()

    
    model_path_best = model_path + f'/kernelborous_poly_{voctype}_end_to_end_lambda_{lambdas[min_err_ind]}'
    save_loc_poly = model_path_best + '/checkpoint_100.tar'
    full_model_poly,full_opt_poly,full_scheduler_poly = load_model(save_loc_poly,kernel_type='full_poly')
    (train_mu,test_mu),(train_sd,test_sd) = eval_model_error(dls,full_model_poly,dt=1/sr,comparison='test')

    data_df = pd.DataFrame({
        'lambdas': lambdas,
        'train MSE': lam_train_cv_err,
        'test MSE': lam_test_cv_err
    })
    data_df.to_csv(os.path.join(model_path,'cv_errs.csv'))        
    """
    return


if __name__ == '__main__':

    fire.Fire(run_model)