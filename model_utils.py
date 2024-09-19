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

            
            x,y = batch # each is bsz x seq len x n neurons + 1
            bsz,_,n = x.shape

            mask_aud = torch.rand(size=(bsz,1)) >= mask_prob_aud
            mask_neur = torch.rand(size=(bsz, 1,n- 1)) >= mask_prob_neur

            x[:,:,-1] *= mask_aud
            x[:,:,:-1] *= mask_neur

            x = x.to('cuda')
            y = y.to('cuda')

            
            yhat = model(x)
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
            train_losses.append(l.item())
        

        if epoch % val_freq == 0:
            model.eval()
            vl = 0.
            for idx,batch in enumerate(loaders['val'],start=epoch*len(loaders['train'])):
                with torch.no_grad():
                    x,y = batch
                    yhat = model(x.to('cuda')) + x.to('cuda')
                    l = loss_fn(y.to('cuda')[:,:,-1],yhat[:,:,-1])
    
                
                    vl += l.item()
            if scheduler:
                scheduler.step(vl/len(loaders['val']))
            val_losses.append((epoch*len(loaders['train']),vl/len(loaders['val'])))
            

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