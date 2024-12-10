from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from tqdm import tqdm
from utils import deriv_approx_d2y,deriv_approx_dy
import matplotlib.pyplot as plt

def train(model,optimizer,loss_fn,loaders,filter=None,scheduler=None,nEpochs=100,val_freq=25,mask_prob_aud = 0.1,init_len = 200,runDir='.',dt=1/44100,vis_freq=100):

    writer = SummaryWriter(log_dir=runDir)

    train_losses,val_losses=[],[]
    
    for epoch in tqdm(range(nEpochs),desc='training model'):

        model.train()

        for idx,batch in enumerate(loaders['train'],start=epoch*len(loaders['train'])):

            optimizer.zero_grad()
            x,y = batch # each is bsz x seq len x n neurons + 1
            bsz,_,n = x.shape

            x = x.to('cuda').to(torch.float64)
            y = y.to('cuda').to(torch.float64)

            
            dy = deriv_approx_dy(x)
            # dy_4dt, dy_3dt, ...., dy_(L-4)dt
            dy2 = deriv_approx_d2y(x)
            # d2y_4dt, d2y_5dt, ..., d2y_(L-4)dt            
            
            yhat,state_pred,penalty = model(x,dt) #state: B x L x SD
            
            yhat = yhat * (model.tau*dt)**2
            
            y = dy2
            L = y.shape[1]

            if (idx % vis_freq) == 0:
                model.visualize(x,dt)
            
                ax = plt.gca()
                ax.plot(yhat[0,:40,0].detach().cpu().numpy())
                ax.plot(y[0,:40,0].detach().cpu().numpy())
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
            alpha = 0 #max(0,min(1,(idx-10*len(loaders['train']))/5000))
            l = loss + alpha*penalty
            #print(l)
            l.backward()
            optimizer.step()
            train_losses.append((loss.item(),penalty.item()))
            writer.add_scalar('Loss/train',loss.item(),idx)
            writer.add_scalar('Penalty/train',penalty.item(),idx)

        if epoch % val_freq == 0:
            model.eval()
            vl = 0.
            vp = 0.
            vn = 0.
            for idx,batch in enumerate(loaders['val'],start=epoch*len(loaders['train'])):
                with torch.no_grad():
                    x,y = batch
                    x = x.to('cuda').to(torch.float64)
                    y = y.to('cuda').to(torch.float64)
                    
                    dy = deriv_approx_dy(y)
                    # dy_4dt, dy_3dt, ...., dy_(L-4)dt
                    dy2 = deriv_approx_d2y(y)
                    # d2y_4dt, d2y_5dt, ..., d2y_(L-4)dt            
                    
                    yhat,state_pred,penalty = model(x,dt) #state: B x L x SD
                    yhat = yhat * (model.tau*dt)**2
                    y = dy2 
                    
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