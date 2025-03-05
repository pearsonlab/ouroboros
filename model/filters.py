import torch
import torch.nn as nn 

def calc_padding_size(filter_size):

    return filter_size//2


class filter(nn.Module):

    def __init__(self,n_filters=4,filter_size=256):

        try:
            iter(n_filters)

        except:
            
            n_filters = [n_filters]
       
        filters = nn.Conv1d(in_channels=1,out_channels=n_filters[0],kernel_size=filter_size)
        filters += [nn.Conv1d(in_channels=i,out_channels=o,kernel_size=filter_size) for i,o in zip(n_filters[:-1],n_filters[1:])]
        filters += nn.Conv1d(in_channels=n_filters[-1],out_channels=1,kernel_size=filter_size)
        self.filter = nn.Sequential(filters)

    def forward(self,x):

        if len(x.shape) == 2:
            x = x[:,None,:]
        return self.filter(x)
    

