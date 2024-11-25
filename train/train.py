from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from tqdm import tqdm
from utils import deriv_approx_d2y,deriv_approx_dy
import matplotlib.pyplot as plt

def train(model,optimizer,loss_fn,loaders,filter=None,scheduler=None,nEpochs=100,val_freq=25,mask_prob_aud = 0.1,init_len = 200,runDir='.',dt=1/44100,vis_freq=100):

    writer = SummaryWriter(log_dir=runDir)

    train_losses,val_losses=[],[]
    #dt_scale = torch.FloatTensor([dt**2])[None,None,:].to('cuda')
    
    for epoch in tqdm(range(nEpochs),desc='training model'):

        model.train()

        for idx,batch in enumerate(loaders['train'],start=epoch*len(loaders['train'])):

            optimizer.zero_grad()
            x,y = batch # each is bsz x seq len x n neurons + 1
            bsz,_,n = x.shape

            x = x.to('cuda').to(torch.float64)
            y = y.to('cuda').to(torch.float64)
            #tx = torch.arange(dt/2,0.3 + dt/2,dt)[:L]
            #sins = torch.sin(2*np.pi*2000*tx).to('cuda')
            #coss = torch.cos(2*np.pi*2000*tx).to('cuda')
            #length = int(round(0.3/dt))
            #print(length)
            #print(L)
            
            dy = deriv_approx_dy(x)
            # dy_2dt, dy_3dt, ...., dy_(L-2)dt
            dy2 = deriv_approx_d2y(x)
            # d2y_4dt, d2y_5dt, ..., d2y_(L-4)dt            
            
            yhat,state_pred,penalty = model(x,dt,idx) #state: B x L x SD
            #print((dt*model.tau)**2)
            yhat = yhat * (model.tau*dt)**2
            #yhat = torch.cat([yhat[:,:-1,:] + dy[:,:-1,:],yhat[:,1:,:]],dim=-1) # one-step prediction too
            y = dy2#torch.cat([dy[:,1:,:],dy2[:,1:,:]],dim=-1)
            L = y.shape[1]

            #sins = torch.sin(2*np.pi*2000*tx).to('cuda')
            if (idx % vis_freq) == 0:
                with torch.no_grad():
                    omega,gamma,b,b_out,d,_ = model.get_funcs(x[:1,:,:],dt)
                    omega,gamma,b,b_out,d = omega.detach().cpu().numpy(),gamma.detach().cpu().numpy(),\
                                        b.detach().cpu().numpy(),\
                                        b_out.detach().cpu().numpy(),\
                                        d.detach().cpu().numpy()
                    torch.cuda.empty_cache()
                on = np.random.choice(L - 45)
                t = np.arange(on,on+40,1)*dt
                ax = plt.gca()
                ax.plot(t,omega[0,on:on+40,0]/ (2*np.pi))
                ax.set_title('omega')
                plt.show()
                plt.close()
                ax = plt.gca()
                ax.plot(t,gamma[0,on:on+40,0] / (2*np.pi))
                ax.set_title('gamma')
                plt.show()
                plt.close()
                ax = plt.gca()
                ax.plot(t,b[0,on:on+40,0]/ (2*np.pi))
                ax.set_title('b')
                plt.show()
                plt.close()
                ax = plt.gca()
                ax.plot(t,d[0,on:on+40,0] / (2*np.pi))
                ax.set_title('d')
                plt.show()
                plt.close()
                #ax = plt.gca()
                #ax.plot(yhat[0,:40,1].detach().cpu().numpy(),label='prediction')
                #ax.plot(y[0,:40,1].detach().cpu().numpy(),label='data')
                #ax.legend()
                #ax.set_title("dy2")
                #plt.show()
                #plt.close() 
                
                ax = plt.gca()
                ax.plot(t,yhat[0,on:on+40,0].detach().cpu().numpy(),label='prediction')
                ax.plot(t,y[0,on:on+40,0].detach().cpu().numpy(),label='data')
                ax.legend()
                ax.set_title("dy2")
                ax.set_xlabel("time (s)")
                plt.show()
                plt.close()
                #assert False
            
            #yhat = torch.cat([dy[:,:-1,:] + yhat[:,:-1,:], yhat[:,1:,:]],dim=-1)
            
            #y = torch.cat([dy[:,1:,:],dy2[:,1:,:]],dim=-1)#torch.cat([y[:,1:,:],dy[:,1:,:],dy2],dim=-1)
            """
            ax = plt.gca()
            ax.plot(yhat[0,:40,0].detach().cpu().numpy())
            ax.plot(y[0,:40,0].detach().cpu().numpy())
            plt.show()
            plt.close()

            ax = plt.gca()
            ax.plot(yhat[0,:40,1].detach().cpu().numpy())
            ax.plot(y[0,:40,1].detach().cpu().numpy())
            plt.show()
            plt.close()

            assert False
            """
            #print(y[0,:3,:])
            #assert False
            ### trend filtering penalty
            #diff = torch.diff(state_pred,dim=1)
            #penalty = torch.abs(diff).sum(dim=-1).mean()

            
            ### sparsity penalty
            alpha = 1/(state_pred.shape[-1] * state_pred.shape[-2])
            norm2 = torch.linalg.vector_norm(state_pred,ord=0.5,dim=-1)
            norm1 = torch.linalg.vector_norm(norm2,ord=1,dim=-1)
            #penalty = alpha*norm1.mean()
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

            
            mse = loss_fn(y,yhat[:,:L,:]) #+ loss_fn(dy[:,2:,:],dyhat)# last channel, which is audio. only want to predict this!
            alpha = max(0,min(1,(idx-10*len(loaders['train']))/5000))
            l = mse + alpha*penalty
            #print(l)
            l.backward()
            optimizer.step()
            train_losses.append((mse.item(),penalty.item()))
            writer.add_scalar('Loss/train',mse.item(),idx)
            writer.add_scalar('Penalty/train',penalty.item(),idx)
            model._clip_weights()
            #writer.add_scalar('State norm/train',norm_penalty.item(),idx)

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
                    # dy_2dt, dy_3dt, ...., dy_(L-2)dt
                    dy2 = deriv_approx_d2y(y)
                    # d2y_4dt, d2y_5dt, ..., d2y_(L-4)dt            
                    
                    yhat,state_pred,penalty = model(x,dt,idx) #state: B x L x SD
                    yhat = yhat * (model.tau*dt)**2
                    #yhat = torch.cat([yhat[:,:-1,:] + dy[:,:-1,:],yhat[:,1:,:]],dim=-1) # one-step prediction too
                    y = dy2 #torch.cat([dy[:,1:,:],dy2[:,1:,:]],dim=-1)
                    #yhat = torch.cat([dy[:,:-1,:] + yhat[:,:-1,:]/dt_scale * dt, yhat[:,1:,:]],dim=-1)
                    
                    #y = torch.cat([dy[:,1:,:],dy2[:,1:,:]],dim=-1)
                    #y = dy2
                    L = y.shape[1]
                    l = loss_fn(y,yhat[:,:L,:])
    
                    vl += l.item()

                    ### trend filtering penalty
                    #diff = torch.diff(state_pred,dim=1)
                    #penalty = torch.abs(diff).sum(dim=-1).mean()
                    
                    #### norm penalty
                    
                    alpha = 1/(state_pred.shape[-1] * state_pred.shape[-2])
                    norm2 = torch.linalg.vector_norm(state_pred,ord=0.5,dim=-1)
                    norm1 = torch.linalg.vector_norm(norm2,ord=1,dim=-1)
                    #penalty = alpha * norm1.mean()
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