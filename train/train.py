from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from tqdm import tqdm
from utils import deriv_approx_d2y,deriv_approx_dy
import matplotlib.pyplot as plt

def eval_model(dls,model,dt):


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

 

def train(model,optimizer,loss_fn,loaders,filter=None,scheduler=None,
          nEpochs=100,val_freq=25,mask_prob_aud = 0.1,
          init_len = 200,runDir='.',dt=1/44100,vis_freq=100,
          use_trend_filtering=False,
          trend_level=1,alpha=1):
    
    print(f'training with trend filtering alpha = {alpha}')

    writer = SummaryWriter(log_dir=runDir)

    train_losses,val_losses=[],[]
    
    for epoch in tqdm(range(nEpochs),desc='training model'):

        model.train()

        for idx,batch in enumerate(loaders['train'],start=epoch*len(loaders['train'])):

            optimizer.zero_grad()
            x,y = batch # each is bsz x seq len x n neurons + 1
            bsz,_,n = x.shape

            x = x.to('cuda').to(torch.float32)
            y = y.to('cuda').to(torch.float32)

            dy = deriv_approx_dy(x)
            # dy_4dt, dy_3dt, ...., dy_(L-4)dt
            #change: scaling to "true" d2y
            dy2 = deriv_approx_d2y(x)/(dt**2)
            # d2y_4dt, d2y_5dt, ..., d2y_(L-4)dt            
            
            y2hat,state_pred,trend_penalty = model(x,dt,use_trend_filtering=use_trend_filtering,trend_level=trend_level) #state: B x L x SD
            
            # change: scaling to "true" d2y
            y2hat = y2hat * model.tau**2 #* (model.tau*dt)**2
            
            #y1hat = dy + y2hat/(dt*model.tau) # makes dy 5:-3
            
            #yhat = x[:,5:-3] + y1hat * (dt*model.tau) # makes y[6:-2]
            
            yhat = y2hat #torch.cat([yhat[:,:-2],y1hat[:,1:-1],y2hat[:,2:]],dim=-1) #points 6:-4
            #print(yhat.shape)
            # y starts as x[1:0]
            y = dy2 #torch.cat([x[:,5:-5],dy[:,2:],dy2[:,2:]],dim=-1)
            L = y.shape[1]

            if vis_freq > 0:
                if (idx % vis_freq) == 0:
                    model.visualize(x,dt)
                
                    on = np.random.choice(L-45)
                    ax = plt.gca()
                    ax.plot(yhat[0,on:on+40,0].detach().cpu().numpy())
                    ax.plot(y[0,on:on+40,0].detach().cpu().numpy())
                    ax.set_title("data (orange) vs model (blue)")
                    plt.show()
                    plt.close()

            """    
            ### trend filtering penalty
            diff = torch.diff(state_pred,dim=1)
            penalty = torch.abs(diff).sum(dim=-1).mean()
            """
            
            """
            ### sparsity penalty
            alpha = 1/(state_pred.shape[-1] * state_pred.shape[-2])
            norm2 = torch.linalg.vector_norm(state_pred,ord=0.5,dim=-1)
            norm1 = torch.linalg.vector_norm(norm2,ord=1,dim=-1)
            penalty = alpha*norm1.mean()
            """

            """
            ### covariance penalty
            cov = state_pred.transpose(-1,-2) @ state_pred /L 
            sds = torch.diagonal(cov,dim1=-1,dim2=-2).sqrt()[:,:,None]
            denom = sds @ sds.transpose(-1,-2)
            abscorr = (cov/denom).abs()
    
            inds = torch.triu_indices(abscorr.shape[-1],abscorr.shape[-1],offset=-1,device=abscorr.device)
            penalty = abscorr[:,inds[0],inds[1]].sum(dim=-1).mean()
            """

            ##################################
            
            loss = loss_fn(y,yhat[:,:L,:]) 
            #alpha = max(0,min(1,(idx-10*len(loaders['train']))/5000)) if use_trend_filtering else 0
            l = loss + alpha*trend_penalty
            #print(l)
            l.backward()
            optimizer.step()
            train_losses.append((loss.item(),trend_penalty.item()))
            writer.add_scalar('Loss/train',loss.item(),idx)
            writer.add_scalar('Penalty/train',trend_penalty.item(),idx)

        if epoch % val_freq == 0:
            model.eval()
            vl = 0.
            vp = 0.
            vn = 0.
            for idx,batch in enumerate(loaders['val'],start=epoch*len(loaders['train'])):
                with torch.no_grad():
                    x,y = batch
                    x = x.to('cuda').to(torch.float32)
                    y = y.to('cuda').to(torch.float32)
                    
                    dy = deriv_approx_dy(y)
                    # dy_4dt, dy_3dt, ...., dy_(L-4)dt
                    #scaling to "true" d2y
                    dy2 = deriv_approx_d2y(y)/(dt**2)
                    # d2y_4dt, d2y_5dt, ..., d2y_(L-4)dt            
                    
                    y2hat,state_pred,penalty = model(x,dt,use_trend_filtering=use_trend_filtering,trend_level=trend_level) #state: B x L x SD
            
                    ## scaling to "true" d2y
                    y2hat = y2hat * model.tau **2 #(model.tau*dt)**2
                    
                    #y1hat = dy + y2hat/(dt*model.tau) # makes dy 5:-3
                    
                    #yhat = x[:,5:-3] + y1hat * (dt*model.tau) # makes y[6:-2]
                    
                    yhat = y2hat #torch.cat([yhat[:,:-2],y1hat[:,1:-1],y2hat[:,2:]],dim=-1) #points 6:-4
                    #print(yhat.shape)
                    # y starts as x[1:0]
                    y = dy2 #torch.cat([x[:,6:-4],dy[:,2:],dy2[:,2:]],dim=-1)
                    
                    L = y.shape[1]
                    l = loss_fn(y,yhat[:,:L,:])
    
                    vl += l.item()

                    """
                    ### trend filtering penalty
                    diff = torch.diff(state_pred,dim=1)
                    penalty = torch.abs(diff).sum(dim=-1).mean()
                    """

                    """
                    #### norm penalty
                    
                    alpha = 1/(state_pred.shape[-1] * state_pred.shape[-2])
                    norm2 = torch.linalg.vector_norm(state_pred,ord=0.5,dim=-1)
                    norm1 = torch.linalg.vector_norm(norm2,ord=1,dim=-1)
                    penalty = alpha * norm1.mean()
                    #vp += alpha*penalty.item()
                    """ 

                    """
                    #### cov penalty

                    cov = state_pred.transpose(-1,-2) @ state_pred /L 
                    sds = torch.diagonal(cov,dim1=-1,dim2=-2).sqrt()[:,:,None]
                    denom = sds @ sds.transpose(-1,-2)
                    abscorr = (cov/denom).abs()
            
                    inds = torch.triu_indices(abscorr.shape[-1],abscorr.shape[-1],offset=-1,device=abscorr.device)
                    penalty = abscorr[:,inds[0],inds[1]].sum(dim=-1).mean()
                    """
                    
                    vp += penalty.item()
                    
                    
            if scheduler:
                scheduler.step(l.item()/len(loaders['val']))
            val_losses.append((epoch*len(loaders['train']),vl/len(loaders['val']),vp/len(loaders['val'])))
            writer.add_scalar('Loss/validation',vl/len(loaders['val']),idx)
            writer.add_scalar('Penalty/validation',vp/len(loaders['val']),idx)

    writer.close()
    return train_losses,val_losses,model,optimizer