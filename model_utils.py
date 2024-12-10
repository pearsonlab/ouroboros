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

def smooth(data,smooth_len):

    B,L,D = data.shape
    pad = torch.zeros((B,smooth_len,D),device='cuda')
    try:
        cumsum = torch.cumsum(torch.cat([pad,data],dim=1),dim=1)
    except:
        print(pad.shape,data.shape)
        assert False
    return (cumsum[:,smooth_len:,:] - cumsum[:,:-smooth_len,:])/float(smooth_len)

class NonNegClipper(object):

    def __init__(self,min=0):
        self.min=0

    def __call__(self,module):

        if hasattr(module,'weight'):
            w = module.weight.data
            w.clamp_(min=self.min)
        if hasattr(module,'bias'):
            b = module.bias.data
            b.clamp_(min=self.min)