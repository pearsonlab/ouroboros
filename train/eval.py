import torch
import numpy as np
from tqdm import tqdm
from utils import deriv_approx_d2y,deriv_approx_dy
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

def integrate_rm(data,rm_length=5):

    """
    fill this in with code from your notebook
    """
    pass 

def eval_model_integration(dls,model,dt,n_segs=100):

    for seg in n_segs:

        train_b = next(iter(dls['train']))
        test_b = next(iter(dls['val']))

        train_x,train_y = train_b
        test_x,test_y = test_b 

        test_dy2,train_dy2 = deriv_approx_d2y(test_x)/(dt**2), deriv_approx_d2y(train_x)/(dt**2)

    


