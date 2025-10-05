from train.train import train,save_model,load_model
from model.kernels import fullPolyModule
from model.constrained_model import rkhs_ouroboros
from utils import sse 
from visualization.model_vis import loss_plot
from train.eval import eval_model_error

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

def model_cv_lambdas(dls,dt,nEpochs=100,lr=1e-3,\
                    n_kernels=15,expand_factor=10,\
                    n_layers=4,d_state=1,d_conv=4,\
                    tau=1000,smooth_len=0.001,\
                    model_path='',save_freq=5):



    model_info={'n layers':n_layers,
                'd state':d_state,
                'd conv':d_conv,
                'expand factor':expand_factor}
    

    #lambdas = np.array([0.01,0.05,0.1,0.25,0.5,0.75,1.]) #* 10**2.5
    min_lambda = 1.01
    max_lambda = 10**(4/(2*n_kernels))
    #lambdas = np.array([1.5,2.0,0.1,0.25,0.5,0.75,1.]) #* 10**2.5
    lambdas = np.linspace(min_lambda,max_lambda,7)
    #im going to ASSUME that with lambda = 1, regularization is 
    lambda_xaxis = np.arange(len(lambdas))
    #n_kernels = 30 #1,2,3
    lam_train_cv_err = []
    lam_test_cv_err = []

    lam_train_cv_sd = []
    lam_test_cv_sd = []
    for ii,lam in enumerate(lambdas):
        
        print(f"Regularizing with lambda={lam}")

        kernel = fullPolyModule(nTerms=n_kernels,device='cuda',x_dim=1,z_dim=2,\
                                activation = lambda x: x,lam=lam,trend_filtering=True)
        reg_weights=True    
        full_model_poly=rkhs_ouroboros(d_data=1,n_layers=n_layers,d_state=d_state,\
                d_conv=d_conv,expand_factor=expand_factor,tau=tau,\
                            smooth_len=smooth_len,kernel=kernel)

        full_opt_poly = Adam(full_model_poly.parameters(),
                    lr=lr)
        full_scheduler_poly = ReduceLROnPlateau(full_opt_poly,factor=0.5,patience=max(nEpochs//25,2),min_lr=1e-10)
        model_path_full_poly = model_path + f'/kernelborous_poly_end_to_end_lambda_{lam}'
        save_loc_poly = model_path_full_poly + f'/checkpoint_{nEpochs}.tar'
        save_files = glob.glob(os.path.join(model_path_full_poly,'*.tar'))

        start_epoch = 0
        if len(save_files) > 0:
            full_model_poly,full_opt_poly,full_scheduler_poly,start_epoch = load_model(model_path_full_poly,kernel_type='full_poly')

        if start_epoch < nEpochs:


            tl,vl,full_model_poly,full_opt_poly = train(full_model_poly,full_opt_poly,loss_fn=lambda y,yhat: sse(yhat,y,reduction='mean'),\
                                loaders=dls,scheduler=full_scheduler_poly,nEpochs=nEpochs,val_freq=1,\
                        runDir=model_path_full_poly,\
                        dt = dt,vis_freq=max(nEpochs//10,1),smoothing=False,\
                            reg_weights=reg_weights,start_epoch=start_epoch,
                            save_freq=save_freq,model_info=model_info)
        
            loss_plot(tl,vl,save_loc=model_path_full_poly,show=False)

            
            save_model(full_model_poly,full_opt_poly,save_loc_poly,
                       n_layers=n_layers,d_state=d_state,
                       expand_factor=expand_factor,d_conv=d_conv)
        
        full_model_poly.eval()
        with torch.no_grad():
            (train_mu,test_mu),(train_sd,test_sd) = eval_model_error(dls,full_model_poly,dt=dt)
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

    
    model_path_best = model_path + f'/kernelborous_poly_end_to_end_lambda_{lambdas[min_err_ind]}'
    #save_loc_poly = model_path_best + '/checkpoint_100.tar'
    full_model_poly,full_opt_poly,full_scheduler_poly,_ = load_model(model_path_best,kernel_type='full_poly')
    full_model_poly.eval()
    with torch.no_grad():
        (train_mu,test_mu),(train_sd,test_sd) = eval_model_error(dls,full_model_poly,dt=dt,comparison='test')

    data_df = pd.DataFrame({
        'lambdas': lambdas,
        'train MSE': lam_train_cv_err,
        'test MSE': lam_test_cv_err
    })
    data_df.to_csv(os.path.join(model_path,'cv_errs.csv'))        
    
    return full_model_poly

