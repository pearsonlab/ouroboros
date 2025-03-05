import torch
import torch.nn as nn 

def filter(nn.Module):

    def __init__(self,n_layers,filter_sizes=8):

        try:
            iter(filter_sizes)

        except:
            filter_sizes = [filter_sizes]