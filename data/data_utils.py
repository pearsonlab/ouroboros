from torch.utils.data import Dataset,DataLoader
import torch 
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.interpolate import make_interp_spline,make_smoothing_spline
from tqdm import tqdm
from scipy.signal import savgol_filter


class aud_neur_ds(Dataset):

    def __init__(self,data):
        self.x = data
        self.dxdt = savgol_filter(self.x,window_length=5,polyorder=3,deriv=1)
        self.dx2dt2 = savgol_filter(self.x,window_length=5,polyorder=3,deriv=2)

    def __len__(self):

        return self.x.shape[0]

    def __getitem__(self, idx):

        x = self.x[idx]
        dxdt = self.dxdt[idx]
        dx2dt2 = self.dx2dt2[idx]

        x,dxdt,dx2dt2 = torch.from_numpy(x).type(torch.DoubleTensor),torch.from_numpy(dxdt).type(torch.DoubleTensor),torch.from_numpy(dx2dt2).type(torch.DoubleTensor)

        return x,dxdt,dx2dt2
    
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

def eval_interp(sample_x,sample_y,lam,n_samples=20):

    choices = np.random.choice(len(sample_x),n_samples,replace=False)
    var,intVar = 0,0
    for c in choices:
        a,t = sample_y[c],sample_x[c]
        spl = make_smoothing_spline(t,a,lam=lam)
        int = spl(t)
        intVar +=  np.nanvar(int)
        var += np.nanvar(a)

    return var / n_samples,intVar/n_samples

def interp_samples(sample_x,sample_y,lam):

    new_y = []
    for y,x in tqdm(zip(sample_x,sample_y)):

        spl = make_smoothing_spline(x,y,lam=lam)
        new_y.append(spl(x))
    
    return np.stack(new_y,axis=0)


def get_loaders_interp(data,num_workers=4,batch_size=32,train_size=0.8,\
                cv = False,seed=None,oversample_prop=1,dt=1/44100,starting_lam=1e-15):

    dls = {}
    if oversample_prop > 1:
        print('oversampling') 
    else:
        pass
    test_size = 1 - train_size
    val_size= test_size/2
    X_train,X_test = train_test_split(data,test_size=test_size,random_state=seed)
    t = np.arange(0,X_train.shape[1]*dt,dt)[:X_train.shape[1]]
    t_train = np.tile(t[None,:],(X_train.shape[0],1))

    X_train = interp_samples(t_train,X_train,lam=starting_lam)

    if cv:
        X_val, X_test = train_test_split(X_test,test_size=0.5,random_state=seed)

        t_val = np.tile(t[None,:],(X_val.shape[0],1))

        X_val = interp_samples(t_val,X_val,lam=starting_lam)
        dsVal = aud_neur_ds(X_val)
        if oversample_prop > 1:
            dsVal.interpolate_oversample(oversample_prop=oversample_prop,dt=dt)
        dls['val'] = DataLoader(dsVal,num_workers=num_workers,batch_size=batch_size,shuffle=False)
    t_test = np.tile(t[None,:],(X_test.shape[0],1))

    X_test = interp_samples(t_test,X_test,lam=starting_lam)
    dsTrain,dsTest = aud_neur_ds(X_train),aud_neur_ds(X_test)
    if oversample_prop > 1:
        dsTrain.interpolate_oversample(oversample_prop=oversample_prop,dt=dt)
        dsTest.interpolate_oversample(oversample_prop=oversample_prop,dt=dt)
    dls['train'] = DataLoader(dsTrain,num_workers=num_workers,batch_size=batch_size,shuffle=True)
    dls['test'] = DataLoader(dsTest,num_workers=num_workers,batch_size=batch_size,shuffle=False)

    return dls