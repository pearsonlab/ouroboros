from torch.utils.data import Dataset,DataLoader
import torch 
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.interpolate import make_interp_spline


class aud_neur_ds(Dataset):

    def __init__(self,data):
        self.data = data

    def __len__(self):

        return self.data.shape[0]

    def __getitem__(self, idx):

        sample = self.data[idx]
        x,y = sample[:-1,:],sample[1:,:]

        x,y = torch.from_numpy(x).type(torch.DoubleTensor),torch.from_numpy(y).type(torch.DoubleTensor)

        return x,y
    
    def interpolate_oversample(self,oversample_prop,dt):

        L = self.data.shape[1]
        new_data = []
        for d in self.data:
            old_t = np.arange(0,L*dt+dt/2,dt)[:L]
            new_t = np.arange(0,L*dt+dt/2,dt/oversample_prop)[:L*oversample_prop]
            spl = make_interp_spline(old_t,d)
            new_data.append(spl(new_t))

        self.data = np.stack(new_data,axis=0)
            

def time_stretch(data,true_dt,fake_dt):

    L = len(data)
    T = fake_dt * L
    currTimes = np.arange(0,T + fake_dt/2,fake_dt)
    newTimes = np.arange(0,T + true_dt/2,true_dt)

    interp = np.interp(newTimes,currTimes,data)

    return interp
    

def euler_integrate(y0,dy,dt):

    return np.cumsum(dy*dt,axis=1) + y0

def adjusted_euler_integrate(y0,dy,d2y,dt=1):

    dy_adjusted = dy*dt + 1/2 * d2y * (dt**2)

    return y0 + np.cumsum(dy_adjusted,axis=1)

def get_loaders(data,num_workers=4,batch_size=32,train_size=0.8,\
                cv = False,seed=None,oversample_prop=1,dt=1/44100):

    dls = {}
    if oversample_prop > 1:
        print('oversampling') 
    else:
        pass
    test_size = 1 - train_size
    val_size= test_size/2
    X_train,X_test = train_test_split(data,test_size=test_size,random_state=seed)


    if cv:
        X_val, X_test = train_test_split(X_test,test_size=0.5,random_state=seed)
        dsVal = aud_neur_ds(X_val)
        if oversample_prop > 1:
            dsVal.interpolate_oversample(oversample_prop=oversample_prop,dt=dt)
        dls['val'] = DataLoader(dsVal,num_workers=num_workers,batch_size=batch_size,shuffle=False)
    dsTrain,dsTest = aud_neur_ds(X_train),aud_neur_ds(X_test)
    if oversample_prop > 1:
        dsTrain.interpolate_oversample(oversample_prop=oversample_prop,dt=dt)
        dsTest.interpolate_oversample(oversample_prop=oversample_prop,dt=dt)
    dls['train'] = DataLoader(dsTrain,num_workers=num_workers,batch_size=batch_size,shuffle=True)
    dls['test'] = DataLoader(dsTest,num_workers=num_workers,batch_size=batch_size,shuffle=False)

    return dls