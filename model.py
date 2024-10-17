from mambapy.mamba import Mamba, MambaConfig

import torch
import torch.nn as nn



def corr_fnc(ts1,tsSet):

    sd1,sd2 = ts1.std(dim=1,keepdim=True), tsSet.std(dim=1,keepdim=True)
    batch_denom = sd1.transpose(2,1) @ sd2

    batch_cov = ts1.transpose(2,1) @ tsSet/(ts1.shape[1]-1)

    batch_corr = batch_denom/batch_cov
    #mean_corrs = batch_corr.mean(dim=0)

    #inds = torch.triu_indices(mean_corrs.shape[0],mean_corrs.shape[1])
    
    return batch_corr.mean(dim=0).sum()


vmap_corr = torch.vmap(corr_fnc,in_dims=(2,None))


class ouroboros(nn.Module):


    def __init__(self,d_data,
                 d_control,
                 d_out=1,
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

        dataConfig = MambaConfig(d_model=d_control,\
                                    n_layers=n_layers_data,d_state=d_state_data,\
                                    d_conv=d_conv_data,expand_factor=expand_factor_data)
        controlConfig = MambaConfig(d_model=d_data*2, n_layers=n_layers_control,\
                                    d_state=d_state_control,d_conv=d_conv_control,\
                                    expand_factor=expand_factor_control)
        
        self.dataMamba = Mamba(dataConfig).to(device)
        self.controlMamba = Mamba(controlConfig).to(device)

        self.outProj = nn.Linear(d_control,d_out).to(device)
        self.control_proj = nn.Linear(d_data*2,d_control).to(device)
        self.device = device


    def forward(self,x,y,mask_ind = -1):

        dy = y - x

        state_pred = self.controlMamba(torch.flip(torch.cat([dy,y],dim=-1),[1]))
        state_pred = torch.flip(torch.nn.SiLU()(self.control_proj(state_pred)),[1]) # bsz x L x dim

        #b,L,d = state_pred.shape

        #state_mus = state_pred.mean(dim=1,keepdim=True)
        #corr_penalty = vmap_corr(state_pred - state_mus, state_pred - state_mus)
        #assert corr_penalty.shape == 

        if 0 <= mask_ind <= x.shape[1]:

            mask = torch.ones(x.shape,device=x.device)
            mask[:,mask_ind,:] = 0
            x *= mask 

        yhat = self.dataMamba(state_pred)
        yhat = self.outProj(yhat)

        ## penalize correlation between states?

        return yhat, state_pred
    
class timescale_ouroboros(ouroboros):


    def __init__(self,d_data,
                 d_control,
                 fs,
                 control_time_constant=0.05, # time constant of control signal in s 
                 n_layers_data=2,
                 n_layers_control=2,
                 d_state_data=16,
                 d_state_control=16,
                 d_conv_data=4,
                 d_conv_control=4,
                 expand_factor_data=1,
                 expand_factor_control=1,
                 device='cuda'):
        
        super(timescale_ouroboros,self).__init__(d_data,
                                                d_control=d_control,
                                                n_layers_data=n_layers_data,
                                                n_layers_control=n_layers_control,
                                                d_state_data=d_state_data,
                                                d_state_control=d_state_control,
                                                d_conv_data=d_conv_data,
                                                d_conv_control=d_conv_control,
                                                expand_factor_data=expand_factor_data,
                                                expand_factor_control=expand_factor_control,
                                                device=device)
        #    d_control,n_layers_data,n_layers_control,\
        #                                        d_state_data,d_state_control,\
        #                                        d_conv_data,d_conv_control,\
        #                                        expand_factor_data,expand_factor_control,\
        #                                        device)

        self.fs = fs
        self.control_time_constant_s = control_time_constant
        self.control_time_constant_fs = fs * control_time_constant
        
        #condition_config  = MambaConfig(d_model=d_data + d_control,\
        #                            n_layers=n_layers_data,d_state=d_state_data,\
        #                            d_conv=d_conv_data,expand_factor=expand_factor_data)
        #controlConfig = MambaConfig(d_model=d_data*2, n_layers=n_layers_control,\
        #                            d_state=d_state_control,d_conv=d_conv_control,\
        #                            expand_factor=expand_factor_control)
        
        #self.dataMamba = Mamba(dataConfig).to(device)
        #self.controlMamba = Mamba(controlConfig).to(device)

        self.control_init_conditions = nn.Linear(d_data*2,d_control).to(device)
        #self.outProj = nn.Linear(d_data + d_control,1).to(device)
        #self.control_proj = nn.Linear(d_data*2,d_control).to(device)


    def forward(self,x,y,mask_ind = -1):

        dy = y - x

        state_ic = torch.nn.SiLU()(self.control_init_conditions(torch.cat([dy[:,:1,:],y[:,:1,:]],dim=-1)))
        assert state_ic.shape[1] == 1,print(state_ic.shape)
        state_pred = self.controlMamba(torch.cat([dy,y],dim=-1))
        state_pred = self.control_proj(state_pred)
        state_pred = state_ic + torch.cumsum(state_pred,dim=1)/self.control_time_constant_fs

        if 0 <= mask_ind <= x.shape[1]:

            mask = torch.ones(x.shape,device=x.device)
            mask[:,mask_ind,:] = 0
            x *= mask 

        yhat = self.dataMamba(torch.cat([state_pred,x],dim=-1))

        ## penalize correlation between states?

        return yhat, state_pred



        
