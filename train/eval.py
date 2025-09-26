import torch
import numpy as np
from tqdm import tqdm
from utils import remove_rm,integrate_d2y,sst,sse,get_spec,deriv_approx_d2y,deriv_approx_dy,butter_filter
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as lr
from itertools import repeat
from model.model_utils import smooth
import os
from torchdiffeq import odeint_adjoint
from scipy.signal import savgol_filter,hilbert
import time
from scipy.interpolate import make_interp_spline
from visualization.model_vis import format_axes

def correct(data,scale_env=False,env_data=[],n_rounds=1):
    corrected = data.copy()
    low_pass = butter_filter(data,cutoff=100,fs=len(data),btype='low')
    corrected = data - low_pass
    #for _ in range(n_rounds):
    #    smoothed = savgol_filter(corrected,window_length=10,polyorder=3)
    #    corrected = corrected - smoothed
    if scale_env:
        assert data.shape == env_data.shape,print(data.shape,env_data.shape)
        x_env = smooth(np.abs(hilbert(env_data))[None,:],smooth_len=10).squeeze()
        c_env = smooth(np.abs(hilbert(corrected))[None,:],smooth_len=10).squeeze()

        corrected *= x_env/c_env
        ## do smoothed envelope stuff here -- probably divide by

    return corrected

def assess_integration_torch(model,x,dt,method='RK45',int_length=0.01,\
                             smoothing=True,strategy='interp',oversample_prop=1,\
                                burn_in_length=0.001,int_start=0,plot=False):

        # general integration params
        B,L,D = x.shape

        smooth_len = int(round(model.smooth_len/dt))
        int_length = min(L*dt, int_length)
        int_length_samples = int(round(int_length/dt))

        burn_in_start = int(round(burn_in_length/dt))
        int_start_samples = int(round(int_start/dt))
        x = x[:,int_start_samples:,:]
    
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
        #print(torch.amax(xdot))
        #print(torch.amin(xdot))
        #print(torch.amax(x),torch.amin(x))
        omega,gamma,weighted_kernels,weights,states = model.get_funcs(x,xdot,dt,scaled=True,smoothing=smoothing)
        
        omega,gamma,weighted_kernels= omega.detach().cpu().numpy().squeeze(),gamma.detach().cpu().numpy().squeeze(),\
                        weighted_kernels.detach().cpu().numpy().squeeze()
        weights = weights.detach().cpu().numpy().reshape([weights.shape[0],weights.shape[1],-1]).squeeze()
        #print(weights.shape)
        #assert False    
        #weights = np.zeros(weights.shape)
        #print(np.amax(weights),np.amin(weights))
        
        t_steps = np.arange(0,L*dt+dt/2,dt)[:L]
        t_eval = torch.arange(0,int_length,dt/oversample_prop,device=model.device)[:int_length_samples*oversample_prop]
        dt_used = dt/oversample_prop

        omegaTerp = make_interp_spline(t_steps,omega)#lambda t: np.interp(t,t_steps,omega)
        gammaTerp = make_interp_spline(t_steps,gamma)#lambda t: np.interp(t,t_steps,gamma)
        weighted_kernelsTerp = lambda t: np.interp(t,t_steps,weighted_kernels)
        weightsTerp = [make_interp_spline(t_steps,weights[:,ii]) for ii in range(weights.shape[1])] #[lambda t,n=ii: np.interp(t,t_steps,weights[:,n]) for ii in range(weights.shape[1])]
        #for ii in range(weights.shape[1]):
        #    print(weights[0,ii],weightsTerp[ii](t_steps[0]))
        #assert False

        z0 = z[0,0,:]
        
        z0[-1] /= dt

        #tau = self.tau#.detach().cpu().numpy()
        xddot_terp = make_interp_spline(t_steps,xddot.detach().cpu().numpy().squeeze()) #lambda t: np.interp(t,t_steps,xddot.detach().cpu().numpy().squeeze())

        def dz_true(t,z):

            s_ind = min(L-1,int(t/dt))
            t= t.detach().cpu().numpy()

            if strategy=='interp':
                dz2_step = torch.from_numpy(np.array([xddot_terp(t)])).to(model.device).to(torch.float32)
            else:
                dz2_step = xddot[0,s_ind,0] #torch.from_numpy(xddot[s_ind]])).to(model.device).to(torch.float32)

            dz1_step = z[1]
            return torch.hstack([dz1_step,dz2_step])
        
        yhat,*_ = model.forward(x[:1,:,:],xdot[:1,:,:],dt)
        
        yhat = yhat.detach().cpu().numpy().squeeze()*model.tau**2
        assert np.sum(np.isnan(yhat)) == 0, print(np.sum(np.isnan(yhat))/np.prod(yhat.shape))
        if plot:
            ax = plt.gca()
            ax.plot(xddot.detach().cpu().numpy().squeeze())
            ax.plot(yhat)
            plt.show()
            
        yhat_terp = make_interp_spline(t_steps,yhat)#lambda t: np.interp(t,t_steps,yhat)
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
        max_dz2 = np.amax(xddot.abs().detach().cpu().numpy().squeeze())
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
                #print(weights_step.shape)
                #print(omega_step.shape)
                #print(gamma_step.shape)

            #print(omegaTerp(t),gammaTerp(t),np.reshape(np.stack([terp(t) for terp in weightsTerp]),[1,1,P,P])[0,0,:,0])
            #print(omega_step,gamma_step,weights_step[0,0,:,0])


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
            #if t/t_eval[-1].detach().cpu().numpy()>0.01: assert False

            dz2 = torch.from_numpy(np.array([-(omega_step**2)*z1 - gamma_step * z2 - weighted_kernels_step])).to(model.device).to(torch.float32).squeeze()
            #dz2 = np
            dz1 = z[1]/dt
            #print(z1,z2)
            #print(dz1,dz2)
            
            #print(-(omega_step**2)*z1)
            #print(-gamma_step*z2)
            #print(weighted_kernels_step)
            return torch.hstack([dz1,dz2])

        print('integrating empirical dz2')
        y1 = odeint_adjoint(dz_true,z0,t_eval,adjoint_params=(),method=method,options = dict()).transpose(0,1)
        print('done! integrating model dz2...')
        y2 = odeint_adjoint(dz_hat1,z0,t_eval,adjoint_params=(),method=method,options = dict()).transpose(0,1)
        print('done! integrating model functions....')
        s = time.time()
        y3 = odeint_adjoint(dz_hat2,z0,t_eval,adjoint_params=(),method=method,options = dict()).transpose(0,1)
        print(f'integrated {int_length:0.2f}s of data in {time.time() - s:0.2f}s')
        #ax = plt.gca()
        #ax.plot()
        #obj1 = solve_ivp(dz_true,(0,int_length),z0.squeeze().detach().cpu().numpy(),t_eval=t_eval,method=method,atol=1e-5)
        #obj2 = solve_ivp(dz_hat1,(0,int_length),z0.squeeze().detach().cpu().numpy(),t_eval=t_eval,method=method,atol=1e-5)
        #obj3 = solve_ivp(dz_hat2,(0,int_length),z0.squeeze().detach().cpu().numpy(),t_eval=t_eval,method=method,atol=1e-5)
        dy1 = y1[1,::oversample_prop].detach().cpu().numpy().squeeze()[burn_in_start:]
        dy2 = y2[1,::oversample_prop].detach().cpu().numpy().squeeze()[burn_in_start:]
        dy3 = y3[1,::oversample_prop].detach().cpu().numpy().squeeze()[burn_in_start:]
        y1 = y1[0,::oversample_prop].detach().cpu().numpy().squeeze()[burn_in_start:]
        y2 = y2[0,::oversample_prop].detach().cpu().numpy().squeeze()[burn_in_start:]
        y3 = y3[0,::oversample_prop].detach().cpu().numpy().squeeze()[burn_in_start:]
        #y1,y2,y3 = obj1.y[0,::oversample_prop],obj2.y[0,::oversample_prop],obj3.y[0,::oversample_prop]
        if plot:
            ax = plt.gca()
            ax.plot(y3)
            ax.set_title("Integrated model, pre correction")
            plt.show()
            plt.close()
            ax = plt.gca()
            ax.plot(y1,label='integrated empirical d2y')
            ax.plot(y2,label='integrated model predicted d2y')
            ax.plot(y3,label='integrated model functions')
            plt.legend()
            ax.set_title("pre correction")
            plt.show()
            #plt.close()
        
        #print(np.sum(np.isnan(y1))/np.prod(y1.shape))
        #print(np.sum(np.isnan(y2))/np.prod(y2.shape))
        #print(np.sum(np.isnan(y3))/np.prod(y3.shape))
        y1_corr,y2_corr,y3_corr = correct(y1),correct(y2),correct(y3)
        #print(np.sum(np.isnan(y1_corr))/np.prod(y1_corr.shape))
        #print(np.sum(np.isnan(y2_corr))/np.prod(y2_corr.shape))
        #print(np.sum(np.isnan(y3_corr))/np.prod(y3_corr.shape))
        if plot:
            ax = plt.gca()
            ax.plot(y3_corr)
            ax.set_title("Integrated model, post correction")
            plt.show()
            plt.close()
            ax = plt.gca()
            ax.plot(y1_corr,label='integrated empirical d2y')
            ax.plot(y2_corr,label='integrated model predicted d2y')
            ax.plot(y3_corr,label='integrated model functions')
            ax.set_title("post correction")
            plt.legend()
            plt.show()
            #plt.close()
        
        xTrue = x.detach().cpu().numpy().squeeze()[:int_length_samples][burn_in_start:]
        y1_corr = y1_corr.squeeze()
        y2_corr = y2_corr.squeeze()
        y3_corr = y3_corr.squeeze()
        trueErr = ((y1_corr - xTrue)**2).sum()
        hat1Err = ((y2_corr - xTrue)**2).sum()
        hat2Err = ((y3_corr - xTrue)**2).sum()
        if plot:
            ax = plt.gca()
            ax.plot(xTrue,label='true')
            ax.plot(y1_corr,label='integrated empirical')
            ax.plot(y3_corr,label='full integration')
            #ax.plot(y2,label='integrated predicted d2')
            plt.legend()
            plt.show()
            #plt.close()
        
        min_len = np.amin(list(map(len,[xTrue,y1_corr,y2_corr,y3_corr])))
        xTrue,y1_corr,y2_corr,y3_corr = xTrue[:min_len],y1_corr[:min_len],y2_corr[:min_len],y3_corr[:min_len]
        truspec,ittr,iftr,*_ = get_spec(xTrue,int(round(1/dt)),onset=0,offset=xTrue.shape[0]*dt,\
                         shoulder=0.0,interp=False,win_len=1028,normalize=False,\
                         min=-2,max=3.5,spec_type='log')
        intspec,*_ = get_spec(y1_corr,int(round(1/dt)),onset=0,offset=y1.shape[0]*dt,\
                         shoulder=0.0,interp=False,win_len=1028,normalize=False,\
                         min=-2,max=3.5,spec_type='log')
        int2spec,*_ = get_spec(y2_corr,int(round(1/dt)),onset=0,offset=y2.shape[0]*dt,\
                         shoulder=0.0,interp=False,win_len=1028,normalize=False,\
                         min=-2,max=3.5,spec_type='log')
        modelspec,*_ = get_spec(y3_corr,int(round(1/dt)),onset=0,offset=y3.squeeze().shape[0]*dt,\
                         shoulder=0.0,interp=False,win_len=1028,normalize=False,\
                         min=-2,max=3.5,spec_type='log')
        

        if plot:                        
            fig,axs = plt.subplots(nrows=1,ncols=4)
            axs[0].imshow(truspec,origin='lower',aspect='auto',extent=[ittr[0],ittr[-1],iftr[0],iftr[-1]])
            axs[1].imshow(intspec,origin='lower',aspect='auto',extent=[ittr[0],ittr[-1],iftr[0],iftr[-1]])
            axs[1].set_yticks([])
            axs[2].imshow(int2spec,origin='lower',aspect='auto',extent=[ittr[0],ittr[-1],iftr[0],iftr[-1]])
            axs[2].set_yticks([])
            axs[3].imshow(modelspec,origin='lower',aspect='auto',extent=[ittr[0],ittr[-1],iftr[0],iftr[-1]])
            axs[3].set_yticks([])
            
            plt.show()
            #plt.close()
        pix_errs_emp = np.linalg.norm(truspec - intspec)
        pix_errs_d2 = np.linalg.norm(truspec - int2spec)
        pix_errs_mod = np.linalg.norm(truspec-modelspec)
        return (trueErr,hat1Err,hat2Err),(y1,y2,y3),\
            (y1_corr,y2_corr,y3_corr),(truspec,intspec,int2spec,modelspec,(ittr,iftr)),\
            (pix_errs_emp,pix_errs_d2,pix_errs_mod)

def eval_model_error(dls,model,dt,comparison='val',return_all=False):

    #tf = model.trend_filtering

    model.eval()

    train_errors = []
    test_errors = []
    preds,reals,data = [],[],[]
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
            
            #vars.append(y.detach().cpu().numpy().flatten())
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
            tot = sst(y,reduction='none') #((y - y.mean(dim=1,keepdim=True))**2).sum(dim=1)
            tot = tot.detach().cpu().numpy().squeeze()
            assert tot.shape == err.shape
            test_r2.append(1 - err/tot)
            reals.append(y.detach().cpu().numpy().squeeze())
            preds.append(dx2hat.detach().cpu().numpy().squeeze())
            #y = y.detach().cpu().numpy()
            data.append(x.detach().cpu().numpy().squeeze())
            test_errors.append(err)

    mean_r2_train = np.nanmean(np.hstack(train_r2))
    mean_r2_test = np.nanmean(np.hstack(test_r2))
    sd_r2_train = np.nanstd(np.hstack(train_r2))
    sd_r2_test = np.nanstd(np.hstack(test_r2))

    print(f"Train r2: {mean_r2_train} +- {sd_r2_train}")
    print(f"{comparison} r2: {mean_r2_test} +- {sd_r2_test}")
    #model.trend_filtering=tf
    if return_all:
        return (mean_r2_train,mean_r2_test),(sd_r2_train,sd_r2_test),\
            (np.hstack(train_r2),np.hstack(test_r2)),(np.vstack(preds),np.vstack(reals),np.vstack(data))
    else:
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

def full_eval_model(model,loaders=None,original_data=None,\
                    dt=1/44100,plot_dir='',use_results=False,
                    n_int=100,plot_steps=False):


    ### Characterize R2 and reconstructions #################
    # removing plots here
    #r2_mosaic = """
    #ABBCCDDEEF
    #AEEG
    #"""
    #fig_r2=plt.figure(layout='constrained',figsize=(15,8))
    #r2_ax_dict=fig_r2.subplot_mosaic(r2_mosaic)
    model.eval()
    if not use_results:
        ### if we've already done this, just load results (maybe from saved csv?) and re-plot
        # implement once the basic stuff is all working
        if loaders == None:
            r2s = []
            predictions =[]
            reals = []
            for o in original_data:
                if len(o.shape) == 1:
                    o = o[None,:,None]
                elif len(o.shape) == 2:
                    o = o.shape[None,:,:]
                #print(o.shape)
                dx2dt2 = torch.from_numpy(deriv_approx_d2y(o)).to(model.device).to(torch.float32)
                dxdt = torch.from_numpy(deriv_approx_dy(o)).to(model.device).to(torch.float32)
                dx2 = dx2dt2/(dt**2)
                x = torch.from_numpy(o).to(model.device).to(torch.float32)
                #print(x.shape)
                #print(dx2.shape)
                #print(dxdt.shape)
                with torch.no_grad():
                    dx2hat,_ = model(x,dxdt,dt,False)
                    dx2hat *= model.tau**2
                    

                    errs = sse(dx2hat,dx2,reduction='sum').detach().cpu().numpy().squeeze()
                    totals = sst(dx2,reduction='sum').detach().cpu().numpy().squeeze()

                    r2 = 1- errs/totals
                r2s.append(r2)

                predictions.append(dx2hat.detach().cpu().numpy().squeeze())
                reals.append(dx2.detach().cpu().numpy().squeeze())

            r2s = np.array(r2s)
            best = np.argmax(r2s)
            worst = np.argmin(r2s)

            best_traj_pred,best_traj_real = predictions[best],reals[best]
            worst_traj_pred,worst_traj_real = predictions[worst],reals[worst]
            
        
        else:

            with torch.no_grad():
                _,_,(_,test_r2s),(test_preds,test_reals,original_data)= eval_model_error(loaders,model,dt,\
                                                                                comparison='test',return_all=True)
            r2s = np.array(test_r2s)
            best = np.argmax(r2s)
            worst = np.argmin(r2s)
            reals = test_reals
            preds = test_preds

            best_traj_pred,best_traj_real = test_preds[best],test_reals[best]
            worst_traj_pred,worst_traj_real = test_preds[worst],test_reals[worst]
            
        
        #r2_ax_dict['A'].boxplot(np.array(r2s),zorder=1)
        #r2_ax_dict['A'].scatter(np.ones(r2s.shape)+np.random.randn(*r2s.shape)*0.1,r2s,s=3,zorder=2)

        #r2_ax_dict['A'].set_ylim([0,1])
        best_res = (best_traj_real - best_traj_pred)*(dt**2)
        worst_res = (worst_traj_real - worst_traj_pred)*(dt**2)

        #r2_ax_dict['B'].plot(best_traj_real*(dt**2))
        #r2_ax_dict['B'].plot(best_traj_pred*(dt**2),alpha=0.3)
        #ylim_b = r2_ax_dict['B'].get_ylim()

        #r2_ax_dict['C'].plot(worst_traj_real*(dt**2))
        #r2_ax_dict['C'].plot(worst_traj_pred*(dt**2),alpha=0.3)
        #ylim_w = r2_ax_dict['C'].get_ylim()

        #r2_ax_dict['D'].plot(best_res)
        #r2_ax_dict['D'].set_ylim(ylim_b)
        #r2_ax_dict['E'].plot(worst_res)
        #r2_ax_dict['E'].set_ylim(ylim_w)

        #r2_ax_dict['F'].hist(best_res,bins=100,density=True)
        sd = np.nanstd(best_res)
        px = lambda x: (1/np.sqrt(2*np.pi*sd**2))*np.exp(-x**2/(2*sd**2))
        #xlims=r2_ax_dict['F'].get_xlim()
        #xax = np.linspace(xlims[0],xlims[1],1000)
        #yax = px(xax)
        #r2_ax_dict['F'].plot(xax,yax,color='tab:red')

        #r2_ax_dict['G'].hist(worst_res,bins=100,density=True)
        sd = np.nanstd(worst_res)
        px = lambda x: (1/np.sqrt(2*np.pi*sd**2))*np.exp(-x**2/(2*sd**2))
        #xlims=r2_ax_dict['G'].get_xlim()
        #xax = np.linspace(xlims[0],xlims[1],1000)
        #yax = px(xax)
        #r2_ax_dict['G'].plot(xax,yax,color='tab:red')
        #for key in r2_ax_dict.keys():
        #    format_axes(r2_ax_dict[key])
        #plt.savefig(os.path.join(plot_dir,'residuals_plot.svg'))
        #plt.close()

        
    #################### Characterize integration ######################

    n_to_integrate = min(n_int,len(reals))
    order = np.random.choice(len(reals),n_to_integrate,replace=False)
    pix_err_real,pix_err_predd2,pix_err_predmod = [],[],[]
    specs_real,specs_int1,specs_int2,specs_int3,bounds = [],[],[],[],[]
    for o in order:
        sample = original_data[o]
        #print(np.amax(o),np.amin(o))
        if len(sample.shape) == 1:
            sample = sample[None,:,None]
        elif len(sample.shape) == 2:
            sample = sample.shape[None,:,:]
        
        with torch.no_grad():
            _,_,_,(truspec,intspec,int2spec,modelspec,(ittr,iftr)),\
            (pix_errs_emp,pix_errs_d2,pix_errs_mod) = assess_integration_torch(model,\
                                                sample,dt,method='rk4',int_length=0.15,\
                                                smoothing=True, strategy='interp',\
                                                oversample_prop=5,burn_in_length=0.001,\
                                                int_start=0,plot=plot_steps)
        pix_err_real.append(pix_errs_emp)
        pix_err_predd2.append(pix_errs_d2)
        pix_err_predmod.append(pix_errs_mod)
        specs_real.append(truspec)
        specs_int1.append(intspec)
        specs_int2.append(int2spec)
        specs_int3.append(modelspec)
        bounds.append((ittr[0],ittr[-1],iftr[0],iftr[-1]))

    #print(len(pix_err_real))
    #print(len(specs_int1))
    #print(len(specs_int2))
    #print(len(specs_int3))
    #print(len(pix_err_predd2))
    #print(len(pix_err_predmod))

    best = np.argmin(pix_err_predd2)
    best_err_d2_ratio = pix_err_predd2[best]/pix_err_real[best]
    best_spec_real = specs_real[best]
    best_spec_int1 = specs_int1[best]
    best_spec_int2 = specs_int2[best]
    best_spec_int3 = specs_int3[best]
    best_ext = bounds[best]
    worst = np.argmax(pix_err_predd2)
    worst_err_d2_ratio = pix_err_predd2[worst]/pix_err_real[worst]
    worst_spec_real = specs_real[worst]
    worst_spec_int1 = specs_int1[worst]
    worst_spec_int2 = specs_int2[worst]
    worst_spec_int3 = specs_int3[worst]
    worst_ext = bounds[worst]
    #print(best,worst)
    #int_mosaic = """
    #ABCDE
    #AFGHI
    #"""
    #fig_int=plt.figure(layout='constrained',figsize=(15,8))
    #int_ax_dict=fig_int.subplot_mosaic(int_mosaic)

    pix_err_real,pix_err_predd2,pix_err_predmod = np.array(pix_err_real),np.array(pix_err_predd2),np.array(pix_err_predmod)
    
    errs = [pix_err_real,pix_err_predd2,pix_err_predmod]
    labels=['Empirical d2','Model predicted d2', 'Model functions']
    #int_ax_dict['A'].boxplot(errs,tick_labels=labels,zorder=1)
    #int_ax_dict['A'].scatter(np.ones(pix_err_real.shape)+np.random.randn(*pix_err_real.shape)*0.1,pix_err_real,zorder=2,s=1)
    #int_ax_dict['A'].scatter(2*np.ones(pix_err_predd2.shape)+np.random.randn(*pix_err_predd2.shape)*0.1,pix_err_predd2,zorder=2,s=1)
    #int_ax_dict['A'].scatter(3*np.ones(pix_err_predmod.shape)+np.random.randn(*pix_err_predmod.shape)*0.1,pix_err_predmod,zorder=2,s=1)
    #int_ax_dict['A'].set_ylim([0,1])
    #int_ax_dict['B'].imshow(best_spec_real,origin='lower',aspect='auto',extent=best_ext)
    #int_ax_dict['C'].imshow(best_spec_int1,origin='lower',aspect='auto',extent=best_ext)
    #int_ax_dict['D'].imshow(best_spec_int2,origin='lower',aspect='auto',extent=best_ext)
    #int_ax_dict['E'].imshow(best_spec_int3,origin='lower',aspect='auto',extent=best_ext)
    #int_ax_dict['D'].set_title(f"Integration spec error ratio to integrated empirical: {best_err_d2_ratio*100:.2f}%")

    #int_ax_dict['F'].imshow(worst_spec_real,origin='lower',aspect='auto',extent=worst_ext)
    #int_ax_dict['G'].imshow(worst_spec_int1,origin='lower',aspect='auto',extent=worst_ext)
    #int_ax_dict['H'].imshow(worst_spec_int2,origin='lower',aspect='auto',extent=worst_ext)
    #int_ax_dict['I'].imshow(worst_spec_int3,origin='lower',aspect='auto',extent=worst_ext)
    #int_ax_dict['G'].set_title(f"Integration spec error ratio to integrated empirical: {worst_err_d2_ratio*100:.2f}%")
    #for key in int_ax_dict.keys():
    #    format_axes(int_ax_dict[key])
    #plt.savefig(os.path.join(plot_dir,'integration_plot.svg'))
    #plt.close()

    traj_xax= np.arange(0,len(best_traj_real)*dt,dt)[:len(best_traj_real)]
    res_xax= np.arange(0,len(best_res)*dt,dt)[:len(best_res)]
    return r2s,((traj_xax,best_traj_real*(dt**2)),(traj_xax,best_traj_pred*(dt**2))),\
            (res_xax,best_res), errs,\
            (best_spec_real,best_spec_int1,best_spec_int2,best_spec_int3),\
            best_ext


        


