import torch
import numpy as np
from tqdm import tqdm
from utils import deriv_approx_d2y,deriv_approx_dy,remove_rm,integrate_d2y
import matplotlib.pyplot as plt


def eval_model_error(dls,model,dt):

    model.eval()

    train_errors = []
    test_errors = []
    for idx,batch in enumerate(dls['train']):
        with torch.no_grad():
            x,y = batch # each is bsz x seq len x n neurons + 1
            bsz,_,n = x.shape

            x = x.to('cuda').to(torch.float32)
            y = y.to('cuda').to(torch.float32)

            dy = deriv_approx_dy(x)
            # dy_4dt, dy_3dt, ...., dy_(L-4)dt
            #change: scaling to "true" d2y
            dy2 = deriv_approx_d2y(x)/(dt**2)
            y2hat,state_pred,trend_penalty = model(x,dt,use_trend_filtering=model.trend_filtering) #state: B x L x SD
            
            # change: scaling to "true" d2y
            y2hat = y2hat * model.tau**2 #* (model.tau*dt)**2
            
            yhat = y2hat
            y = dy2
            L = y.shape[1]

            se = (y - yhat[:,:L,:])**2
            train_errors.append(se.detach().cpu().numpy().flatten())

    for idx,batch in enumerate(dls['val']):
        with torch.no_grad():
            x,y = batch
            x = x.to('cuda').to(torch.float32)
            y = y.to('cuda').to(torch.float32)
            
            dy = deriv_approx_dy(y)
            # dy_4dt, dy_3dt, ...., dy_(L-4)dt
            #scaling to "true" d2y
            dy2 = deriv_approx_d2y(y)/(dt**2)
            # d2y_4dt, d2y_5dt, ..., d2y_(L-4)dt            
            
            y2hat,state_pred,penalty = model(x,dt,use_trend_filtering=model.trend_filtering) #state: B x L x SD
    
            ## scaling to "true" d2y
            y2hat = y2hat * model.tau **2 #(model.tau*dt)**2
            
            yhat = y2hat
            # y starts as x[1:0]
            y = dy2 
            
            L = y.shape[1]
            se = (y - yhat[:,:L,:])**2
            test_errors.append(se.detach().cpu().numpy().flatten())

    test_errors = np.hstack(test_errors)
    test_mu,test_sd = np.nanmean(test_errors),np.nanstd(test_errors)
    train_errors = np.hstack(train_errors)
    train_mu,train_sd = np.nanmean(train_errors),np.nanstd(train_errors)
    return (train_mu,test_mu),(train_sd,test_sd)



def eval_model_integration(dls,model,dt,n_segs=100):

    max_segs = min(min(len(dls['train'],len(dls['val']))),n_segs)

    train_choices = np.random.choice(len(dls['train']),max_segs,replace=False)
    test_choices = np.random.choice(len(dls['val']),max_segs,replace=False)
    for train_ind,test_ind in zip(train_choices,test_choices):

        train_b = dls['train'].data[train_ind]
        test_b = dls['val'].data[test_ind]

        train_x,train_y = train_b
        test_x,test_y = test_b 
        train_x,test_x = train_x.view(1,len(train_x),1),test_x.view(1,len(test_x),1)
        _,L,_ = train_x.shape

        test_dy2,train_dy2 = deriv_approx_d2y(test_x)/(dt**2), deriv_approx_d2y(train_x)/(dt**2)
        test_dy,train_dy = deriv_approx_dy(test_x)/(dt), deriv_approx_dy(train_x)/(dt)
        z0_test,z0_train = np.hstack([test_x[0,0,0],test_dy[0,0,0]]),np.hstack([train_x[0,0,0],train_dy[0,0,0]])
        t = np.arange(0,len(test_x)*dt + dt/2,dt)[:L]

        test_yint = remove_rm(integrate_d2y(test_dy2,t_samples=t,init_cond=z0_test))
        train_yint = remove_rm(integrate_d2y(train_dy2,t_samples=t,init_cond=z0_train))
    


