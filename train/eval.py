import torch
import numpy as np
from tqdm import tqdm
from utils import remove_rm,integrate_d2y,sst,sse,get_spec #deriv_approx_d2y,deriv_approx_dy
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as lr
from itertools import repeat
from model.model_utils import smooth
import os
from torchdiffeq import odeint_adjoint
from scipy.signal import savgol_filter,hilbert
import time


def correct(data,scale_env=False,env_data=[],n_rounds=1):
    corrected = data.copy()
    
    for _ in range(n_rounds):
        smoothed = savgol_filter(corrected,window_length=10,polyorder=3)
        corrected = corrected - smoothed
    if scale_env:
        assert data.shape == env_data.shape,print(data.shape,env_data.shape)
        x_env = smooth(np.abs(hilbert(env_data))[None,:],smooth_len=10).squeeze()
        c_env = smooth(np.abs(hilbert(corrected))[None,:],smooth_len=10).squeeze()

        corrected *= x_env/c_env
        ## do smoothed envelope stuff here -- probably divide by

    return corrected

def deriv_approx_dy(data):

    return savgol_filter(data,window_length=5,polyorder=3,deriv=1,axis=1)
def deriv_approx_d2y(data):

    return savgol_filter(data,window_length=5,polyorder=3,deriv=2,axis=1)

def assess_integration_torch(model,x,dt,method='RK45',int_length=0.05,\
                             smoothing=True,strategy='interp',\
                                oversample_prop=1,burn_in_length=0.001,\
                                    save_dir = '.'):

        # general integration params
        B,L,D = x.shape

        smooth_len = int(round(model.smooth_len/dt))
        int_length = min(L*dt, int_length)
        int_length_samples = int(round(int_length/dt))

        burn_in_start = int(round(burn_in_length/dt))
    
        ### set up x, y
        # x: x_0, x_dt, x_2dt,...
        xdot= torch.from_numpy(deriv_approx_dy(x)).to(model.device).to(torch.float32)

        #print(xdot[:10])
        #print(xdot[:10]/dt)
        # dx: dx_4dt,dx_5dt,dx_6dt,..., dx_(l-4)dt
        xddot = torch.from_numpy(deriv_approx_d2y(x)).to(model.device).to(torch.float32)/dt**2
        sample_ax = np.arange(0,xddot.shape[1]*dt + dt/2,dt)[:xddot.shape[1]]
        #xddot_hat = make_interp_spline(sample_ax,xddot.detach().cpu().numpy().squeeze())

        x = torch.from_numpy(x).to(model.device).to(torch.float32)
        z = torch.cat([x,xdot],dim=-1)
        L = z.shape[1]
        

        #### get functions from model (assumes we only get one x at a time
        omega,gamma,weighted_kernels,weights,states = model.get_funcs(x,xdot,dt,scaled=True,smoothing=smoothing)
        
        omega,gamma,weighted_kernels= omega.detach().cpu().numpy().squeeze(),gamma.detach().cpu().numpy().squeeze(),\
                        weighted_kernels.detach().cpu().numpy().squeeze()
        weights = weights.detach().cpu().numpy().reshape([weights.shape[0],weights.shape[1],-1]).squeeze()
        #print(weights.shape)
        #weights = np.zeros(weights.shape)
        #print(np.amax(weights),np.amin(weights))
        """
        ax = plt.gca()
        ax.plot(omega)
        ax.plot(gamma)
        plt.show()
        plt.close()
        ax = plt.gca()
        for w in weights.T:
            ax.plot(w)
        plt.show()
        plt.close()
        assert False
        """
        t_steps = np.arange(0,L*dt+dt/2,dt)[:L]
        t_eval = torch.arange(0,int_length,dt/oversample_prop,device=model.device)[:int_length_samples*oversample_prop]
        dt_used = dt/oversample_prop

        omegaTerp = lambda t: np.interp(t,t_steps,omega)
        gammaTerp = lambda t: np.interp(t,t_steps,gamma)
        weighted_kernelsTerp = lambda t: np.interp(t,t_steps,weighted_kernels)
        weightsTerp = [lambda t: np.interp(t,t_steps,weights[:,ii]) for ii in range(weights.shape[1])]

        z0 = z[0,0,:]
        
        z0[-1] /= dt

        #tau = self.tau#.detach().cpu().numpy()
        xddot_terp = lambda t: np.interp(t,t_steps,xddot.detach().cpu().numpy().squeeze())

        def dz_true(t,z):

            s_ind = min(L-1,int(t/dt))
            t= t.detach().cpu().numpy()

            if strategy=='interp':
                dz2_step = torch.from_numpy(np.array([xddot_terp(t)])).to(model.device).to(torch.float32)
            else:
                dz2_step = torch.from_numpy(np.array([xddot[s_ind]])).to(model.device).to(torch.float32)

            dz1_step = z[1]
            return torch.hstack([dz1_step,dz2_step])
        
        yhat,*_ = model.forward(x[:1,:,:],xdot[:1,:,:],dt)
        yhat = yhat.detach().cpu().numpy().squeeze()*model.tau**2
        yhat_terp = lambda t: np.interp(t,t_steps,yhat)
        #print(yhat.shape)
        #print(np.sum(np.isnan(yhat)))
        #print(yhat.shape)
    
        def dz_hat1(t,z):
            s_ind = min(L-1,int(t/dt))

            t= t.detach().cpu().numpy()
            if strategy=='interp':
                dz2_step = torch.from_numpy(np.array([yhat_terp(t)])).to(model.device).to(torch.float32)
            else:
                dz2_step = torch.from_numpy(np.array([yhat[s_ind]])).to(model.device).to(torch.float32)

            dz1_step = z[1]
            return torch.hstack([dz1_step,dz2_step])

        P = model.kernel.nTerms + 1
        
        def dz_hat2(t,z):

            # t: time, should have a timestep of roughly dt. treat as ZOH
            # z: B x 2d
            print(f"{t/t_eval[-1]*100:0.3f}%,",end='\r')
            b_ind = min(int(t/dt),L-1)
            t= t.detach().cpu().numpy()
            if strategy == 'interp':
                omega_step = omegaTerp(t) 
                gamma_step = gammaTerp(t)
                weights_step = np.reshape(np.stack([terp(t) for terp in weightsTerp]),[1,1,P,P])
            else:
                omega_step= omega[b_ind]
                gamma_step = gamma[b_ind]
                weights_step=np.reshape(weights[b_ind],(1,1,P,P))


            z1 = z[:1].detach().cpu().numpy()
            
            z2 = z[1:].detach().cpu().numpy() 
            #print(z2,z)
            dz1 = z[1]

            z[-1] *= dt
            #print(z2,z)
            #print(z.shape)
            #-(omega**2)*z1 - gamma * z2 - weighted_kernels
            weighted_kernels_step = model.kernel.forward_given_weights_numpy(z[None,:].detach().cpu().numpy(),weights_step).squeeze()
            #print(omega_step,gamma_step,weighted_kernels_step)
            #print(z1,z2)
            #if t/t_eval[-1].detach().cpu().numpy()>0.05: assert False

            dz2 = torch.from_numpy(np.array([-(omega_step**2)*z1 - gamma_step * z2 - weighted_kernels_step])).to(model.device).to(torch.float32).squeeze()
            dz1 = z[1]/dt
            #print(z1,z2)
            #print(-(omega_step**2)*z1)
            #print(-gamma_step*z2)
            #print(weighted_kernels_step)
            return torch.hstack([dz1,dz2])

        print('integrating empirical dz2')
        y1 = odeint_adjoint(dz_true,z0,t_eval,adjoint_params=(),method=method,options = dict()).transpose(0,1)
        print('done! integrating model dz2...')
        y2 = odeint_adjoint(dz_hat1,z0,t_eval,adjoint_params=(),method=method,options = dict()).transpose(0,1)
        print('done! integrating model functions....')
        #s = time.time()
        #y3 = odeint_adjoint(dz_hat2,z0,t_eval,adjoint_params=(),method=method,options = dict()).transpose(0,1)
        #print(f'integrated {int_length:0.2f}s of data in {time.time() - s:0.2f}s')
        #ax = plt.gca()
        #ax.plot()
        
        
        dy1 = y1[1,::oversample_prop].detach().cpu().numpy().squeeze()[burn_in_start:]
        dy2 = y2[1,::oversample_prop].detach().cpu().numpy().squeeze()[burn_in_start:]
        #dy3 = y3[1,::oversample_prop].detach().cpu().numpy().squeeze()[burn_in_start:]
        y1 = y1[0,::oversample_prop].detach().cpu().numpy().squeeze()[burn_in_start:]
        y2 = y2[0,::oversample_prop].detach().cpu().numpy().squeeze()[burn_in_start:]
        #y3 = y3[0,::oversample_prop].detach().cpu().numpy().squeeze()[burn_in_start:]
        
    
    
        y1_corr,y2_corr = correct(y1),correct(y2)#,y3_corr,correct(y3)
        
        xTrue = x.detach().cpu().numpy().squeeze()[:int_length_samples]
        trueErr = ((y1_corr - xTrue[burn_in_start:])**2).sum()
        hat1Err = ((y2_corr - xTrue[burn_in_start:])**2).sum()
        #hat2Err = ((y3_corr - xTrue[burn_in_start:])**2).sum()

        ##### y plot ####
        ax = plt.gca()
        ax.plot(xTrue,label='true')
        ax.plot(y1_corr,label='integrated, corrected empirical')
        ax.plot(y2_corr,label='integrated,corrected predicted 2nd derivative')
        #ax.plot(y3_corr,label='full integration')
        #ax.plot(y2,label='integrated predicted d2')
        plt.legend()
        plt.savefig(os.path.join(save_dir,'integrations_y.svg'))
        #plt.show()
        plt.close()

        ##### dy plot #######
        dxTrue = xdot.detach().cpu().numpy().squeeze()[:int_length_samples][burn_in_start:]/dt
        ax = plt.gca()
        ax.plot(dxTrue,label='empirical 1st derivative')
        ax.plot(dy1,label='integrated, corrected empirical')
        ax.plot(dy2,label='integrated,corrected predicted 2nd derivative')
        #ax.plot(y3_corr,label='full integration')
        #ax.plot(y2,label='integrated predicted d2')
        plt.legend()
        plt.savefig(os.path.join(save_dir,'integrations_dy.svg'))
        #plt.show()
        plt.close()

        ##### spectrograms  ######
        truspec,ittr,iftr,*_ = get_spec(xTrue.squeeze(),int(round(1/dt)),onset=0,offset=xTrue.shape[0]*dt,\
                         shoulder=0.0,interp=False,win_len=1028,normalize=False,\
                         min=-2,max=3.5,spec_type='log')
        intspec,*_ = get_spec(y1_corr.squeeze(),int(round(1/dt)),onset=0,offset=y1.squeeze().shape[0]*dt,\
                         shoulder=0.0,interp=False,win_len=1028,normalize=False,\
                         min=-2,max=3.5,spec_type='log')
        int2spec,*_ = get_spec(y2_corr.squeeze(),int(round(1/dt)),onset=0,offset=y2.squeeze().shape[0]*dt,\
                         shoulder=0.0,interp=False,win_len=1028,normalize=False,\
                         min=-2,max=3.5,spec_type='log')
        #modelspec,*_ = get_spec(y3_corr.squeeze(),int(round(1/dt)),onset=0,offset=y3.squeeze().shape[0]*dt,\
        #                 shoulder=0.0,interp=False,win_len=1028,normalize=False,\
        #                 min=-2,max=3.5,spec_type='log')
        
                                    
        fig,axs = plt.subplots(nrows=1,ncols=3)
        axs[0].imshow(truspec,origin='lower',aspect='auto',extent=[ittr[0],ittr[-1],iftr[0],iftr[-1]])
        axs[0].set_title("Original data")
        axs[1].imshow(intspec,origin='lower',aspect='auto',extent=[ittr[0],ittr[-1],iftr[0],iftr[-1]])
        axs[1].set_yticks([])
        axs[1].set_title("Integrated, \n corrected empirical")
        axs[2].imshow(int2spec,origin='lower',aspect='auto',extent=[ittr[0],ittr[-1],iftr[0],iftr[-1]])
        axs[2].set_yticks([])
        axs[2].set_title("Integrated, corrected\n predicted 2nd derivative")
        #axs[3].imshow(modelspec,origin='lower',aspect='auto',extent=[ittr[0],ittr[-1],iftr[0],iftr[-1]])
        #axs[3].set_yticks([])
        
        #plt.show()
        plt.savefig(os.path.join(save_dir,'integrated_spectrograms.svg'))
        plt.close()
        return (trueErr,hat1Err),(y1,y2),(y1_corr,y2_corr) #hat2Err,y3,y3_corr

def eval_model_error(dls,model,dt,comparison='val'):

    #tf = model.trend_filtering

    model.eval()

    train_errors = []
    test_errors = []
    vars = []
    train_r2 = []
    test_r2 = []
    #model.trend_filtering=False

    for idx,batch in enumerate(dls['train']):
        with torch.no_grad():
            x,dxdt,dx2dt2 = batch # each is bsz x seq len x n neurons + 1
            bsz,_,n = x.shape

            x = x.to('cuda').to(torch.float32)
            dxdt = dxdt.to('cuda').to(torch.float32)
            dx2 = dx2dt2.to('cuda').to(torch.float32)/(dt**2)
            dx2hat,state_pred = model(x,dxdt,dt) #state: B x L x SD
            
            # change: scaling to "true" d2y
            dx2hat = dx2hat * model.tau**2 #* (model.tau*dt)**2
            
            yhat = dx2hat
            y = dx2
            L = y.shape[1]

            err = sse(yhat,y,reduction='none')#((y - yhat[:,:L,:])**2).sum(dim=1)
            err = err.detach().cpu().numpy().squeeze()
            #y = y.detach().cpu().numpy()
            tot = sst(y,reduction='none')
            tot = tot.detach().cpu().numpy().squeeze()
            #mse = np.nanmean(se,axis=1)
            train_r2.append(1 - err/tot)
            
            vars.append(y.detach().cpu().numpy().flatten())
            train_errors.append(err)
            

    for idx,batch in enumerate(dls[comparison]):
        with torch.no_grad():
            x,dxdt,dx2dt2 = batch # each is bsz x seq len x n neurons + 1
            bsz,_,n = x.shape

            x = x.to('cuda').to(torch.float32)
            dxdt = dxdt.to('cuda').to(torch.float32)
            dx2 = dx2dt2.to('cuda').to(torch.float32)/(dt**2)
            dx2hat,state_pred = model(x,dxdt,dt) #state: B x L x SD
            
            # change: scaling to "true" d2y
            dx2hat = dx2hat * model.tau**2 #* (model.tau*dt)**2
            
            yhat = dx2hat
            # y starts as x[1:0]
            y = dx2 
            
            L = y.shape[1]
            err = sse(yhat,y,reduction='none') #((y - yhat[:,:L,:])**2).sum(dim=1)
            err = err.detach().cpu().numpy().squeeze()
            #y = y.detach().cpu().numpy()
            tot = sst(y,reduction='none') #((y - y.mean(dim=1,keepdim=True))**2).sum(dim=1)
            tot = tot.detach().cpu().numpy().squeeze()
            assert tot.shape == err.shape
            test_r2.append(1 - err/tot)
            vars.append(y.detach().cpu().numpy().flatten())
            test_errors.append(err)

    mean_r2_train = np.nanmean(np.hstack(train_r2))
    mean_r2_test = np.nanmean(np.hstack(test_r2))
    sd_r2_train = np.nanstd(np.hstack(train_r2))
    sd_r2_test = np.nanstd(np.hstack(test_r2))

    print(f"Train r2: {mean_r2_train} +- {sd_r2_train}")
    print(f"{comparison} r2: {mean_r2_test} +- {sd_r2_test}")
    #model.trend_filtering=tf
    
    return (mean_r2_train,mean_r2_test),(sd_r2_train,sd_r2_test)


def assess_kernels(dataloader,model,dt,saveDir=''):

    ypts = np.linspace(-1,1,25)
    dypts = np.linspace(-1/dt,1/dt,25)/2
    ygrid,dygrid = np.meshgrid(ypts,dypts)
    ygrid = ygrid.flatten()
    dygrid = dygrid.flatten()
    ydy = np.stack([ygrid,dygrid],axis=1)[None,:,:]
    L = ydy.shape[1]

    kernels = []
    smooth_len = int(round(model.smooth_len/dt))

    weight_samples = np.random.choice(len(dataloader),max(len(dataloader)//5 + 1,1),replace=False)
    #print(f"taking samples from batches {weight_samples}")

    for ii,batch in enumerate(dataloader):
        with torch.no_grad():
            x,dxdt,_ = batch
            B,_,_  = x.shape
            #print(x.shape)
            x = x.to('cuda').to(torch.float32)
            dxdt = dxdt.to('cuda').to(torch.float32)
            _,_,_,weights,_ = model.get_funcs(x,dxdt,dt,scaled=True,smoothing=True)

            #weights = smooth(weights,smooth_len)

        weights = weights.detach().cpu().numpy().squeeze()
        normed_weights = weights/np.sum(weights,axis=-1,keepdims=True)
        weights_sample = np.nanmean(normed_weights,axis=1,keepdims=True)
        weights_sample = np.tile(weights_sample,(1,L,1))

        ydy_batch = np.tile(ydy,(B,1,1))
        assert weights_sample.shape[0] == ydy_batch.shape[0], print(weights_sample.shape,ydy_batch.shape)
        assert weights_sample.shape[1] == ydy_batch.shape[1], print(weights_sample.shape,ydy_batch.shape)

        kern = model.kernel.forward_given_weights_numpy(ydy_batch,weights_sample)

        kern = np.reshape(kern,(B,25,25))
        kern = np.sign(kern)*np.log(np.abs(kern) + 1e-12)

        kernels.append(kern)
        if ii in weight_samples:
            batch_sample = np.random.choice(B)
            fig = plt.figure(figsize=(20,5))
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)

            tw,twn = weights[batch_sample],normed_weights[batch_sample]
            for channel in tw.T:
                ax1.plot(channel)

            for channel in twn.T:
                ax2.plot(channel)

            ax1.set_title(f"weights across batch {ii} sample {batch_sample}")
            ax2.set_title("Weights across time, normed across weight channels")

            plt.savefig(os.path.join(saveDir,f'kernel_weights_{ii}_{batch_sample}.png'))
            plt.close()

    kernels = np.vstack(kernels)
    kernels = np.nanmean(kernels,axis=0)


    ax = plt.gca()
    g = ax.imshow(kernels.T,origin='lower',aspect='auto',extent=[ypts[0],ypts[-1],dypts[0],dypts[-1]])
    ax.set_xlabel("y")
    ax.set_ylabel("dy")
    cb = plt.colorbar(g,ax=ax)
    cb.set_label('log(abs(Kernel value)) * sign(kernel value)',rotation=270,labelpad=40)

    #plt.show()
    plt.savefig(os.path.join(saveDir, 'average_kernel_dl.svg'))
    plt.close()

    return kernels


def pad_with_nan(l,target_len):

    l1 = len(l)

    diff = target_len - l1
    if diff > 0:
        return np.hstack([l, np.nan * np.ones(diff,)])
    else:
        return l
    
def eval_model_integration(dls,model,dt,n_segs=100,st=0.05,comparison='val'):

    #tf = model.trend_filtering
    #model.trend_filtering=False

    start = int(round(st/dt))
    max_segs = min(min(len(dls['train'].dataset),len(dls[comparison].dataset)),n_segs)

    train_choices = np.random.choice(len(dls['train'].dataset),max_segs,replace=False)
    test_choices = np.random.choice(len(dls[comparison].dataset),max_segs,replace=False)

    train_errs,test_errs=[],[]
    for train_ind,test_ind in tqdm(zip(train_choices,test_choices),desc='Integrating samples...',total=max_segs):

        train_b = dls['train'].dataset[train_ind]
        test_b = dls[comparison].dataset[test_ind]

        train_x,train_dxdt,train_dx2dt2 = train_b
        test_x,test_dxdt,test_dx2dt2 = test_b 
        train_dx2 = train_dx2dt2.view(1,len(train_dx2dt2),1)/(dt**2)
        test_dx2 = test_dx2dt2.view(1,len(test_dx2dt2),1)/(dt**2)
        train_x,test_x = train_x.view(1,len(train_x),1),test_x.view(1,len(test_x),1)
        train_dxdt,test_dxdt = train_dxdt.view(1,len(train_dxdt),1),test_dxdt.view(1,len(test_dxdt),1)
        
        test_y_modelint,*_ = model.integrate(test_x.to('cuda').to(torch.float32),\
                                             test_dxdt.to('cuda').to(torch.float32),\
                                             test_dx2.to('cuda').to(torch.float32),\
                                                dt,with_residual=False,method='RK45',st=0.0,smoothing=True)
        train_y_modelint,*_ = model.integrate(train_x.to('cuda').to(torch.float32),\
                                              train_dxdt.to('cuda').to(torch.float32),\
                                              train_dx2.to('cuda').to(torch.float32),\
                                                dt,with_residual=False,method='RK45',st=0.0,smoothing=True)
        #test_dy2,train_dy2 = deriv_approx_d2y(test_x)/(dt**2), deriv_approx_d2y(train_x)/(dt**2)
        #test_dy2,train_dy2 = test_dy2.detach().cpu().numpy().squeeze(),train_dy2.detach().cpu().numpy().squeeze()
        #test_dy,train_dy = deriv_approx_dy(test_x)/(dt), deriv_approx_dy(train_x)/(dt)
        #test_dy,train_dy = test_dy.detach().cpu().numpy().squeeze(),train_dy.detach().cpu().numpy().squeeze()
        
        train_x,test_x = train_x.detach().cpu().numpy().squeeze(),test_x.detach().cpu().numpy().squeeze()
        #train_y, test_y = train_y.detach().cpu().numpy().squeeze()[3:-5],test_y.detach().cpu().numpy().squeeze()[3:-5]
        L = len(train_x)
        L_train,L_test = min(L,train_y_modelint.shape[1]),min(L,test_y_modelint.shape[1])
        
        z0_test,z0_train = np.hstack([test_x[0],test_dxdt[0]/dt]),np.hstack([train_x[0],train_dxdt[0]/dt])
        t = np.arange(0,len(test_dxdt)*dt + dt/2,dt)[:L]
        #print(test_dy2.shape)
        #print(test_dy.shape)
        #print(z0_test.shape)

        test_yint = remove_rm(integrate_d2y(test_dx2,t_samples=t,init_cond=z0_test),rm_length=5)[0,start:L_test]
        train_yint = remove_rm(integrate_d2y(train_dx2,t_samples=t,init_cond=z0_train),rm_length=5)[0,start:L_train]
        train_y_modelint = remove_rm(train_y_modelint.squeeze(),rm_length=5)[0,start:L_train]
        test_y_modelint = remove_rm(test_y_modelint.squeeze(),rm_length=5)[0,start:L_test]
        #train_y,test_y = train_y[start:L_train],test_y[start:L_test]
        train_x,test_x = train_x[start:L_train],test_x[start:L_test]
        train_errs.append(np.abs(train_y_modelint - train_x))
        test_errs.append(np.abs(test_y_modelint - test_x))

    
    #max_len_errs = max(list(map(len,train_errs)) + list(map(len,test_errs)))
    #t_dummy = np.arange(0,max_len_errs + 1/2,1)[:max_len_errs]
    #pad = lambda x: pad_with_nan(x,max_len_errs)
    #train_errs = list(map(pad,train_errs))
    #test_errs = list(map(pad,test_errs))
    train_errs = [np.log(e + 1e-10) for e in train_errs]
    #train_errs = np.log(np.stack(train_errs) + 1e-10)
    #train_t_dummy = np.tile(t_dummy[None,:],(max_segs,1))
    test_errs = [np.log(e + 1e-10) for e in test_errs]
    #test_errs = np.log(np.stack(test_errs) + 1e-10)
    #test_t_dummy = np.tile(t_dummy[None,:],(max_segs,1))

    train_coefs,test_coefs = [],[]
    for ty in train_errs:
        L = len(ty)
        tx = np.arange(0,L+1/2,1)[:L]
        lr_train = lr().fit(tx.flatten()[:,None],ty.flatten())
        train_coefs.append(lr_train.coef_)
    for ty in test_errs:
        L = len(ty)
        tx = np.arange(0,L+1/2,1)[:L]
        lr_test = lr().fit(tx.flatten()[:,None],ty.flatten())
        test_coefs.append(lr_test.coef_)
    
    mu_train_coef,mu_test_coef = np.nanmean(train_coefs),np.nanmean(test_coefs)
    sd_train_coef,sd_test_coef = np.nanstd(train_coefs),np.nanstd(test_coefs)

    print(f"Train integration error slope: {mu_train_coef} +- {sd_train_coef}")
    print(f"{comparison} integration error slope: {mu_test_coef} +- {sd_test_coef}")
    #model.trend_filtering=tf

    return (mu_train_coef,mu_test_coef),(sd_train_coef,sd_test_coef)

