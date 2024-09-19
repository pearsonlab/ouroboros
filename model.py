from mambapy.mamba import Mamba, MambaConfig

import torch
import torch.nn as nn

class ouroboros(nn.Module):


    def __init__(self,d_data,
                 d_control,
                 n_layers_data=2,
                 n_layers_control=2,
                 d_state_data=16,
                 d_state_control=16,
                 d_conv_data=4,
                 d_conv_control=4,
                 expand_factor_data=1,
                 expand_factor_control=1,
                 device='cuda'):
        
        super(ouroboros,self).__init__()

        dataConfig = MambaConfig(d_model=d_data + d_control,\
                                    n_layers=n_layers_data,d_state=d_state_data,\
                                    d_conv=d_conv_data,expand_factor=expand_factor_data)
        controlConfig = MambaConfig(d_model=d_data*2, n_layers=n_layers_control,\
                                    d_state=d_state_control,d_conv=d_conv_control,\
                                    expand_factor=expand_factor_control)
        
        self.dataMamba = Mamba(dataConfig).to(device)
        self.controlMamba = Mamba(controlConfig).to(device)

        self.outProj = nn.Linear(d_data + d_control,1).to(device)
        self.control_proj = nn.Linear(d_data*2,d_control).to(device)


    def forward(self,x,y,mask_ind = -1):

        dy = y - x

        state_pred = self.controlMamba(torch.cat([dy,y],dim=-1))
        state_pred = torch.nn.SiLU()(self.control_proj(state_pred))

        if 0 <= mask_ind <= x.shape[1]:

            mask = torch.ones(x.shape,device=x.device)
            mask[:,mask_ind,:] = 0
            x *= mask 

        yhat = self.dataMamba(torch.cat([state_pred,x],dim=-1))

        ## penalize correlation between states?

        return yhat, state_pred



        
