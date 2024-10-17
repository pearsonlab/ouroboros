import torch
from torch.utils.data import Dataset,DataLoader

import numpy as np
from tqdm import tqdm

class aud_neur_ds(Dataset):

    def __init__(self,data):
        self.data = data

    def __len__(self):

        return self.data.shape[0]

    def __getitem__(self, idx):

        sample = self.data[idx]
        x,y = sample[:-1,:],sample[1:,:]

        x,y = torch.from_numpy(x).type(torch.FloatTensor),torch.from_numpy(y).type(torch.FloatTensor)

        return x,y
    
def train(model,optimizer,loss_fn,loaders,filter=None,scheduler=None,nEpochs=100,val_freq=25,mask_prob_aud = 0.1,mask_prob_neur=0.2):

    mask_prob_neur = 2*mask_prob_aud
    gen = np.random.default_rng()
    
    train_losses,val_losses=[],[]
    if filter: 
        print('filtering model output')
        vmapped_filter = torch.vmap(filter,in_axis=3,out_axis=3)
    for epoch in tqdm(range(nEpochs),desc='training model'):

        model.train()

        for idx,batch in enumerate(loaders['train'],start=epoch*len(loaders['train'])):

            optimizer.zero_grad()
            x,y = batch # each is bsz x seq len x n neurons + 1
            bsz,L,n = x.shape

            mask_aud = torch.rand(size=(bsz,1)) >= mask_prob_aud
            mask_neur = torch.rand(size=(bsz, 1,n- 1)) >= mask_prob_neur

            x[:,:,-1] *= mask_aud
            x[:,:,:-1] *= mask_neur

            x = x.to('cuda')
            y = y.to('cuda')

            
            yhat,state_pred = model(x,y) #state: B x L x SD
            cov = state_pred.transpose(-1,-2) @ state_pred /L 
            sds = torch.diagonal(cov,dim1=-1,dim2=-2).sqrt()[:,:,None]
            denom = sds @ sds.transpose(-1,-2)
            abscorr = (cov/denom).abs()
    
            #print(abscorr.shape)
            inds = torch.triu_indices(abscorr.shape[-1],abscorr.shape[-1],offset=-1,device=abscorr.device)
            penalty = abscorr[:,inds[0],inds[1]].sum(dim=-1).mean()

            



            ### predict da/dt
            yhat = yhat + x
            ################

            ###################
            
            ##### learn filter on output ####
            if filter:
                yhat = vmapped_filter(yhat[:,None,:,:]).squeeze()[:,:y.shape[1],:]
                
            
            ##################################

            l = loss_fn(y[:,:,-1],yhat[:,:,-1]) # last channel, which is audio. only want to predict this!
            
            l.backward()
            optimizer.step()
            train_losses.append((l.item(),penalty.item()))
        

        if epoch % val_freq == 0:
            model.eval()
            vl = 0.
            vp = 0.
            for idx,batch in enumerate(loaders['val'],start=epoch*len(loaders['train'])):
                with torch.no_grad():
                    x,y = batch
                    x,y = x.to('cuda'),y.to('cuda')
                    yhat,state_pred = model(x,y)
                    yhat = yhat + x
                    l = loss_fn(y[:,:,-1],yhat[:,:,-1])
    
                
                    vl += l.item()

                    cov = state_pred.transpose(-1,-2) @ state_pred /L 
                    sds = torch.diagonal(cov,dim1=-1,dim2=-2).sqrt()[:,:,None]
                    denom = sds @ sds.transpose(-1,-2)
                    abscorr = (cov/denom).abs()
            
                    inds = torch.triu_indices(abscorr.shape[-1],abscorr.shape[-1],offset=-1,device=abscorr.device)
                    penalty = abscorr[:,inds[0],inds[1]].sum(dim=-1).mean()
                    vp += penalty.item()
            if scheduler:
                scheduler.step(vl/len(loaders['val']))
            val_losses.append((epoch*len(loaders['train']),vl/len(loaders['val']),vp/len(loaders['val'])))
            

    return train_losses,val_losses,model,optimizer

def save(model,opt,save_loc):
    sd = {'model': model.state_dict(),
      'opt':opt.state_dict()}
    
    torch.save(sd,save_loc)

def load(model,opt,load_loc):

    checkpoint = torch.load(load_loc, weights_only=True)

    model.load_state_dict(checkpoint['model'])
    opt.load_state_dict(checkpoint['opt'])

    return model,opt

def simulate_step(model,sample,mask_audio=True):

    caches = [(None, torch.zeros((1,model.config.d_model * model.config.expand_factor,model.config.d_conv),device='cuda')) for _ in model.layers]

    sample = sample.squeeze()
    N = sample.shape[0]

    gen_data = []
    
    for ii in range(N):
        s = sample[ii:ii+1]
        
        if mask_audio:
            mask = torch.ones(s.shape,device=s.device)
            mask[:,-1] = 0
            s = s * mask
        dy,caches = model.step(s,caches)
        gen_data.append(s + dy)

    return torch.vstack(gen_data)[None,:,-1]

def generate(model,sample,mask_audio=False):

    caches = [(None, torch.zeros((1,model.config.d_model * model.config.expand_factor,model.config.d_conv),device='cuda')) for _ in model.layers]
    sample = sample.squeeze()

    N = sample.shape[0]

    gen_data = []
    
    for ii in range(N):
        if ii == 0:
            s = sample[ii:ii+1]
        else:
            s = gen_data[ii-1]
        
        if mask_audio:
            mask = torch.ones(s.shape,device=s.device)
            mask[:,-1] = 0
            s = s * mask
        dy,caches = model.step(s,caches)
        gen_data.append(s + dy)
    
    return torch.vstack(gen_data)[None,:,-1]