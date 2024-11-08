from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from tqdm import tqdm


def train(model,optimizer,loss_fn,loaders,filter=None,scheduler=None,nEpochs=100,val_freq=25,mask_prob_aud = 0.1,init_len = 200,runDir='.',dt=1/44100):

    writer = SummaryWriter(log_dir=runDir)

    train_losses,val_losses=[],[]
    dt_scale = torch.FloatTensor([dt**2])[None,None,:].to('cuda')
    
    for epoch in tqdm(range(nEpochs),desc='training model'):

        model.train()

        for idx,batch in enumerate(loaders['train'],start=epoch*len(loaders['train'])):

            optimizer.zero_grad()
            x,y = batch # each is bsz x seq len x n neurons + 1
            bsz,L,n = x.shape

            x = x.to('cuda')
            y = y.to('cuda')
            
            dy = y - x
            dy2 = torch.diff(dy,dim=1)

            
            yhat,state_pred = model(x,y) #state: B x L x SD
            yhat = yhat * dt_scale
            
            yhat = torch.cat([dy + yhat, yhat],dim=-1)
            y = torch.cat([dy[:,1:,:],dy2],dim=-1)#torch.cat([y[:,1:,:],dy[:,1:,:],dy2],dim=-1)

            ### trend filtering penalty
            #diff = torch.diff(state_pred,dim=1)
            #penalty = torch.abs(diff).sum(dim=-1).mean()

            
            ### sparsity penalty
            alpha = 1/(state_pred.shape[-1] * state_pred.shape[-2])
            norm2 = torch.linalg.vector_norm(state_pred,ord=0.5,dim=-1)
            norm1 = torch.linalg.vector_norm(norm2,ord=1,dim=-1)
            penalty = alpha*norm1.mean()
            """
            ### covariance penalty
            cov = state_pred.transpose(-1,-2) @ state_pred /L 
            sds = torch.diagonal(cov,dim1=-1,dim2=-2).sqrt()[:,:,None]
            denom = sds @ sds.transpose(-1,-2)
            abscorr = (cov/denom).abs()
    
            #print(abscorr.shape)
            inds = torch.triu_indices(abscorr.shape[-1],abscorr.shape[-1],offset=-1,device=abscorr.device)
            penalty = abscorr[:,inds[0],inds[1]].sum(dim=-1).mean()
            """

            ##################################

            
            mse = loss_fn(y,yhat[:,:-1,:]) #+ loss_fn(dy[:,2:,:],dyhat)# last channel, which is audio. only want to predict this!
            l = mse #+ penalty
            l.backward()
            optimizer.step()
            train_losses.append((mse.item(),penalty.item()))
            writer.add_scalar('Loss/train',mse.item(),idx)
            writer.add_scalar('Penalty/train',penalty.item(),idx)
            #writer.add_scalar('State norm/train',norm_penalty.item(),idx)

        if epoch % val_freq == 0:
            model.eval()
            vl = 0.
            vp = 0.
            vn = 0.
            for idx,batch in enumerate(loaders['val'],start=epoch*len(loaders['train'])):
                with torch.no_grad():
                    x,y = batch
                    x = x.to('cuda')
                    y = y.to('cuda')
                    
                    dy = y - x
                    dy2 = torch.diff(dy,dim=1)

                    
                    yhat,state_pred = model(x,y) #state: B x L x SD
                    yhat = yhat * dt_scale
                    
                    yhat = torch.cat([dy + yhat, yhat],dim=-1)
                    y = torch.cat([dy[:,1:,:],dy2],dim=-1)
                    
                    l = loss_fn(y,yhat[:,:-1,:])
    
                    vl += l.item()

                    ### trend filtering penalty
                    #diff = torch.diff(state_pred,dim=1)
                    #penalty = torch.abs(diff).sum(dim=-1).mean()
                    
                    #### norm penalty
                    
                    alpha = 1/(state_pred.shape[-1] * state_pred.shape[-2])
                    norm2 = torch.linalg.vector_norm(state_pred,ord=0.5,dim=-1)
                    norm1 = torch.linalg.vector_norm(norm2,ord=1,dim=-1)
                    penalty = alpha * norm1.mean()
                    #vp += alpha*penalty.item()
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
            #writer.add_scalar('State norm/validation',vn/len(loaders['val']),idx)

    writer.close()
    return train_losses,val_losses,model,optimizer