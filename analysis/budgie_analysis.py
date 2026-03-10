from scipy.io import loadmat,wavfile
from scipy.ndimage import gaussian_filter
from scipy.interpolate import make_interp_spline
import gc
import pickle 
from tqdm import tqdm
from scipy.interpolate import make_interp_spline
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import RidgeCV as R
from sklearn.linear_model import LassoLarsCV as LCV
from sklearn.linear_model import ElasticNetCV as ECV
import glob
import os
import torch
from utils import deriv_approx_dy,deriv_approx_d2y
import numpy as np
import matplotlib.pyplot as plt



def load_spikes_functions(datapath:str,model:object,padding:float =0.1,model_smooth:float=0.001,spike_smooth:float=0.005):
    """
    Docstring for load_spikes_functions
    
    :datapath: string, path to budgie data
    :model: nn.module: ouroboros model, used to obtain model functions
    :padding: float, padding on either end of the audio. expand to give more context for model function initial conditions
    :model_smooth: float, width of filter used to smooth model functions
    :spike_smooth: float, bin width for spikes
    """


    wavpath = glob.glob(os.path.join(datapath,'*_cleaned.wav'))[0]
    segpath = glob.glob(os.path.join(datapath,'*.txt'))[0]
    spikepath = glob.glob(os.path.join(datapath,'*.mat'))[0]

    print('loading audio...')
    sr,audio = wavfile.read(wavpath)
    dt = 1/sr
    L,T = len(audio),len(audio)/sr
    spike_bin_size=0.001
    
    print('loading spikes...')
    spiketimes = loadmat(spikepath)['spikeTimes'][0]
    #print(len(spiketimes))
    #print(spiketimes[0].shape)
    print('loading segments...')
    onoffs = np.loadtxt(segpath,usecols=(0,1))
    print('done!')


    old_smooth_len = model.smooth_len
    model.smooth_len = model_smooth
    bin_sd_smooth = int(round(spike_smooth/spike_bin_size))
    bin_func_smooth = int(round(spike_smooth*sr))

    voc_spikes = []
    voc_omegas = []
    voc_gammas = []
    voc_rates= []
    
    for onset,offset in tqdm(onoffs,desc='getting neural and behavioral reps'):
        gc.collect()

        aud_on,aud_off = int(round(onset * sr)),int(round(offset*sr))

        on_pad = min(aud_on,int(round(padding*sr)))
        off_pad = min(L-aud_off,int(round(padding*sr)))

        x = audio[aud_on-on_pad:aud_off+off_pad][None,:,None]
        dxdt = deriv_approx_dy(x)

        x,dxdt = torch.from_numpy(x).to(model.device).to(torch.float32),torch.from_numpy(dxdt).to(model.device).to(torch.float32)
        with torch.no_grad():
            omega,gamma,*_ = model.get_funcs(x,dxdt,dt,smoothing=False)
        omega,gamma=omega.detach().cpu().numpy().squeeze(),gamma.detach().cpu().numpy().squeeze()
        omega,gamma= gaussian_filter(omega,sigma=bin_func_smooth),gaussian_filter(gamma,sigma=bin_func_smooth)
        omega,gamma = omega[on_pad:],gamma[on_pad:]
        if off_pad > 0:
            omega,gamma= omega[:-off_pad],gamma[:-off_pad]
        
        spikebins = np.arange(onset - on_pad/sr,offset+off_pad/sr + spike_bin_size/2,spike_bin_size)
        spike_on_pad = int(round(on_pad/(sr* spike_bin_size)))
        spike_off_pad = int(round(off_pad/(sr*spike_bin_size)))

        spikes = []
        rates = []
        for s in spiketimes:
            
            binned_spikes,_ = np.histogram(s[(s>=(onset - on_pad/sr))*(s <= (offset+off_pad/sr))],bins=spikebins)
            r = gaussian_filter(binned_spikes/spike_bin_size,sigma=bin_sd_smooth)

            spikes.append(binned_spikes)
            rates.append(r)
        
        spikes = np.vstack(spikes)
        spikes = spikes[:,spike_on_pad:]
        rates = np.vstack(rates)
        rates = rates[:,spike_on_pad:]
        if spike_off_pad > 0:
            spikes = spikes[:,:-spike_off_pad]
            rates = rates[:,:-spike_off_pad]

        voc_rates.append(rates)
        voc_spikes.append(spikes)
        voc_omegas.append(omega)
        voc_gammas.append(gamma)

    gc.collect()
    return voc_omegas,voc_gammas,voc_spikes,voc_rates


def spike_function_design(datapath:str,model:object,model_smooth:float=0.001,spike_smooth:float=0.005,win_length:int=20,seed:int=92):
    """
    Docstring for spike_function_design
    
    :datapath: string, path to Budgie data
    :param model: nn.module, ouroboros model, used to get model functions
    :param model_smooth: float, width of window used to smooth model functions
    :param spike_smooth: float, bin width for spikes
    :param win_length: int, number of spike bins to consider
    :param seed: int or None, seed used for reproducibility
    """

    gen = np.random.default_rng(seed=seed)

    wavpath = glob.glob(os.path.join(datapath,'*_cleaned.wav'))[0]
    segpath = glob.glob(os.path.join(datapath,'*.txt'))[0]
    spikepath = glob.glob(os.path.join(datapath,'*.mat'))[0]

    print('loading audio...')
    sr,audio = wavfile.read(wavpath)
    dt = 1/sr
    L,T = len(audio),len(audio)/sr
    spike_bin_size=0.001
    
    print('loading spikes...')
    spiketimes = loadmat(spikepath)['spikeTimes'][0]
    
    print('loading segments...')
    onoffs = np.loadtxt(segpath,usecols=(0,1))
    print('done!')


    old_smooth_len = model.smooth_len
    model.smooth_len = model_smooth
    bin_sd_smooth = int(round(spike_smooth/spike_bin_size))
    bin_func_smooth = int(round(spike_smooth*sr))

    voc_spikes = []
    voc_omegas = []
    voc_gammas = []
    voc_nulls = []

    padding = win_length * spike_bin_size
    
    for onset,offset in tqdm(onoffs,desc='getting neural and behavioral reps'):
        gc.collect()

        aud_on,aud_off = int(round(onset * sr)),int(round(offset*sr))

        on_pad = min(aud_on,int(round(padding*sr)))
        off_pad = min(L-aud_off,int(round(padding*sr)))

        x = audio[aud_on-on_pad:aud_off+off_pad][None,:,None]
        dxdt = deriv_approx_dy(x)

        x,dxdt = torch.from_numpy(x).to(model.device).to(torch.float32),torch.from_numpy(dxdt).to(model.device).to(torch.float32)

        spike_design =  []
        null_design = []
        with torch.no_grad():
            omega,gamma,*_ = model.get_funcs(x,dxdt,dt,smoothing=False)
        
        
        omega,gamma=omega.detach().cpu().numpy().squeeze(),gamma.detach().cpu().numpy().squeeze()
        omega,gamma= gaussian_filter(omega,sigma=bin_func_smooth),gaussian_filter(gamma,sigma=bin_func_smooth)
        omega,gamma = omega[on_pad:],gamma[on_pad:]
        if off_pad > 0:
            omega,gamma= omega[:-off_pad],gamma[:-off_pad]
        omega_t = np.linspace(onset,offset,len(omega))
        spike_t = np.arange(onset,offset + spike_bin_size/2,spike_bin_size)
    
        omega_spl,gamma_spl = make_interp_spline(omega_t,omega),make_interp_spline(omega_t,gamma)
        omega,gamma = omega_spl(spike_t),gamma_spl(spike_t)
        spikebins = np.arange(onset - on_pad/sr,offset+off_pad/sr + spike_bin_size/2,spike_bin_size)
        spike_on_pad = int(round(on_pad/(sr* spike_bin_size)))
        spike_off_pad = int(round(off_pad/(sr*spike_bin_size)))

        spikes = []
        null_spikes = []
        for s in spiketimes:
            
            binned_spikes,_ = np.histogram(s[(s>=(onset - on_pad/sr))*(s <= (offset+off_pad/sr))],bins=spikebins)
            

            spikes.append(binned_spikes)
            
            shuffle = gen.choice(len(binned_spikes),len(binned_spikes),replace=False)
            null_spikes.append(binned_spikes[shuffle])
        
        spikes = np.vstack(spikes)
        null_spikes = np.vstack(null_spikes)

        if spike_off_pad > 0:
            spikes = spikes[:,:-spike_off_pad]
            null_spikes = null_spikes[:,:-spike_off_pad]
        
        ons = np.arange(0,spikes.shape[1]-win_length)
        offs = ons + win_length
        for on, off in zip(ons,offs):
            #rate_design.append(rates[:,on:off])
            spike_design.append(spikes[:,on:off])
            null_design.append(null_spikes[:,on:off])
        spike_design = np.stack(spike_design,axis=0)
        null_design = np.stack(null_design,axis=0)
        omega = omega[:len(spike_design)]
        gamma = gamma[:len(spike_design)]

        voc_nulls.append(null_design)
        voc_spikes.append(spike_design)
        voc_omegas.append(omega)
        voc_gammas.append(gamma)
        
    gc.collect()

    return voc_omegas,voc_gammas,voc_spikes,voc_nulls


def train_sliding_window_regression(datapath,model,model_smooth=0.001,spike_smooth=0.005,win_length=20,save_path='',seed=92,train_prop=0.5,n_runs=10,n_vocs_vis = 3):
    """
    Docstring for train_sliding_window_regression
    
    :param datapath: string, path to data
    :param model: nn.Module, ouroboros modekl
    :param model_smooth: float, smooth length for model functions
    :param spike_smooth: float, bin size for spikes
    :param win_length: number of spike bins to consider for linear model
    :param save_path: string: save path for data and figures
    :param seed: int or none, for reproducibility
    :param train_prop: float, between 0-1, amount of data used to train linear model
    :param n_runs: int, number of models to train
    :param n_vocs_vis: int, number of vocalizations to visualize parameters for
    """

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    gen = np.random.default_rng(seed=seed)

    model_path_template = os.path.join(save_path,'models_{run:n}.pkl')
    
    data_path = os.path.join(save_path,'data.pkl')

    weight_mats_omega = []
    weight_mats_gamma = []
    r2s_omega,r2s_gamma = {'train':[],'test': []},{'train':[],'test': []}
    r2s_omega_null,r2s_gamma_null = {'train':[],'test': []},{'train':[],'test': []}
    r2s_omega_hardnull,r2s_gamma_hardnull = {'train':[],'test': []},{'train':[],'test': []}
    for run in range(1,n_runs+1):
        print(f"run {run}")
        model_path=model_path_template.format(run=run)
        if not os.path.isfile(model_path):
            gc.collect()
            if not os.path.isfile(data_path):
                print('loading data...')
                omegas,gammas,spike_designs,null_designs = spike_function_design(datapath,model,\
                                                                                 model_smooth=model_smooth,\
                                                                                 spike_smooth=spike_smooth,win_length=win_length,seed=seed)
    
                data_dict = {'spikes':spike_designs,
                         'nulls':null_designs,
                         'omegas':omegas,
                         'gammas':gammas}
        
                with open(data_path,'wb') as f:
                    pickle.dump(data_dict,f)
            else:
                print('loading data...')
                with open(data_path,'rb') as f:
                    data_dict = pickle.load(f)
        
                spike_designs = data_dict['spikes']
                null_designs = data_dict['nulls']
                omegas = data_dict['omegas']
                gammas = data_dict['gammas']
            
            N,T = spike_designs[0].shape[1],win_length
            spike_designs_stacked = np.vstack([x.reshape(x.shape[0],-1) for x in spike_designs])
            null_designs_stacked = np.vstack([x.reshape(x.shape[0],-1) for x in null_designs])
            omegas_stacked,gammas_stacked = np.hstack(omegas),np.hstack(gammas)
        
            
            #train_prop=0.8
            n_train = int(round(train_prop*len(spike_designs_stacked)))
            order = gen.choice(len(spike_designs_stacked),len(spike_designs_stacked),replace=False)
            train_inds,test_inds = order[:n_train],order[n_train:]
            
            train_x,test_x = spike_designs_stacked[train_inds],spike_designs_stacked[test_inds]
            train_null,test_null = null_designs_stacked[train_inds],null_designs_stacked[test_inds]
                        
            train_om,test_om = omegas_stacked[train_inds],omegas_stacked[test_inds]
            train_gam,test_gam = gammas_stacked[train_inds],gammas_stacked[test_inds]

            train_order_hardnull = gen.choice(len(train_om),len(train_om),replace=False)
            test_order_hardnull = gen.choice(len(test_om),len(test_om),replace=False)
            
            train_om_hardnull,test_om_hardnull = train_om[train_order_hardnull],test_om[test_order_hardnull]
            train_gam_hardnull,test_gam_hardnull = train_om[train_order_hardnull],test_om[test_order_hardnull]
        
            print("training omega (real)...")
            om_lr = LCV().fit(train_x,train_om)
            print("training gamma (real)...")
            gam_lr = LCV().fit(train_x,train_gam)
        
            print("training omega (easy null)...")
            om_lr_null= LCV().fit(train_null,train_om)
            print("training gamma (easy null)")
            gam_lr_null = LCV().fit(train_null,train_gam)

            print("training omega (hard null)...")
            om_lr_hardnull= LCV().fit(train_x,train_om_hardnull)
            print("training gamma (hard null)")
            gam_lr_hardnull = LCV().fit(train_x,train_gam_hardnull)
    
        
            train_r2_om,train_r2_gam = om_lr.score(train_x,train_om),gam_lr.score(train_x,train_gam)
        
            test_r2_om,test_r2_gam = om_lr.score(test_x,test_om),gam_lr.score(test_x,test_gam)
        
            train_r2_om_null,train_r2_gam_null = om_lr_null.score(train_null,train_om),gam_lr_null.score(train_null,train_gam)
        
            test_r2_om_null,test_r2_gam_null = om_lr_null.score(test_null,test_om),gam_lr_null.score(test_null,test_gam)

            train_r2_om_hardnull,train_r2_gam_hardnull = om_lr_hardnull.score(train_null,train_om_hardnull),gam_lr_hardnull.score(train_x,train_gam_hardnull)
        
            test_r2_om_hardnull,test_r2_gam_hardnull = om_lr_hardnull.score(test_x,test_om_hardnull),gam_lr_hardnull.score(test_x,test_gam_hardnull)
        
            model_dict = {'real':{'omega':{'model':om_lr,'train r2':train_r2_om,'test r2': test_r2_om},
                                 'gamma':{'model':gam_lr,'train r2':train_r2_gam,'test r2': test_r2_gam}},
                          'null':{'omega':{'model': om_lr_null,'train r2':train_r2_om_null,'test r2': test_r2_om_null},
                                 'gamma':{'model': gam_lr_null,'train r2': train_r2_gam_null,'test r2': test_r2_gam_null}},
                          'hard null':{'omega':{'model': om_lr_hardnull,'train r2':train_r2_om_hardnull,'test r2': test_r2_om_hardnull},
                                 'gamma':{'model': gam_lr_hardnull,'train r2': train_r2_gam_hardnull,'test r2': test_r2_gam_hardnull}}
                         }
            with open(model_path,'wb') as f:
                pickle.dump(model_dict,f)
        else:
            print('loading data and models...')
            
            with open(model_path,'rb') as f:
                model_dict = pickle.load(f)
            
            om_lr,train_r2_om,test_r2_om = model_dict['real']['omega']['model'],model_dict['real']['omega']['train r2'], model_dict['real']['omega']['test r2']
            gam_lr,train_r2_gam,test_r2_gam = model_dict['real']['gamma']['model'],model_dict['real']['gamma']['train r2'], model_dict['real']['gamma']['test r2']
    
            om_lr_null,train_r2_om_null,test_r2_om_null = model_dict['null']['omega']['model'],model_dict['null']['omega']['train r2'], model_dict['null']['omega']['test r2']
            gam_lr_null,train_r2_gam_null,test_r2_gam_null = model_dict['null']['gamma']['model'],model_dict['null']['gamma']['train r2'], model_dict['null']['gamma']['test r2']

            om_lr_hardnull,train_r2_om_hardnull,test_r2_om_hardnull = model_dict['hard null']['omega']['model'],model_dict['hard null']['omega']['train r2'], model_dict['hard null']['omega']['test r2']
            gam_lr_hardnull,train_r2_gam_hardnull,test_r2_gam_hardnull = model_dict['hard null']['gamma']['model'],model_dict['hard null']['gamma']['train r2'], model_dict['hard null']['gamma']['test r2']
            if run == 1:
                with open(data_path,'rb') as f:
                    data_dict = pickle.load(f)
        
                spike_designs = data_dict['spikes']
                null_designs = data_dict['nulls']
                omegas = data_dict['omegas']
                gammas = data_dict['gammas']
        
                N,T = spike_designs[0].shape[1],win_length

        print(f"Omega performance train: {train_r2_om:.4f}, test: {test_r2_om:.4f}")
        print(f"Gamma performance train: {train_r2_gam:.4f}, test: {test_r2_gam:.4f}")
    
        print(f"Omega null performance train: {train_r2_om_null:.4f}, test: {test_r2_om_null:.4f}")
        print(f"Gamma null performance train: {train_r2_gam_null:.4f}, test: {test_r2_gam_null:.4f}")

        print(f"Omega hard null performance train: {train_r2_om_hardnull:.4f}, test: {test_r2_om_hardnull:.4f}")
        print(f"Gamma hard null performance train: {train_r2_gam_hardnull:.4f}, test: {test_r2_gam_hardnull:.4f}")
    
        weights_gam = gam_lr.coef_.reshape(N,T)
        weights_om = om_lr.coef_.reshape(N,T)

        weight_mats_omega.append(weights_om)
        weight_mats_gamma.append(weights_gam)
        
        r2s_omega['train'].append(train_r2_om)
        r2s_gamma['train'].append(train_r2_gam)
        r2s_omega_null['train'].append(train_r2_om_null)
        r2s_gamma_null['train'].append(train_r2_gam_null)
        r2s_omega_hardnull['train'].append(train_r2_om_hardnull)
        r2s_gamma_hardnull['train'].append(train_r2_gam_hardnull)

        r2s_omega['test'].append(test_r2_om)
        r2s_gamma['test'].append(test_r2_gam)
        r2s_omega_null['test'].append(test_r2_om_null)
        r2s_gamma_null['test'].append(test_r2_gam_null)
        r2s_omega_hardnull['test'].append(test_r2_om_hardnull)
        r2s_gamma_hardnull['test'].append(test_r2_gam_hardnull)
    
        max_gam = np.amax(np.abs(weights_gam))
        max_om = np.amax(np.abs(weights_om))
        max_max = max(max_gam,max_om)
        
        ax = plt.gca()
        g=ax.matshow(weights_gam,aspect='auto',vmin=-max_max,vmax=max_max,cmap='RdBu_r',vmin=-0.01,vmax=0.01)
        ax.set_ylabel("Neuron number")
        ax.set_xlabel("Lag (ms)")
        ax.set_xticks(range(0,win_length,win_length//4))
        ax.set_xticklabels(range(-win_length,0,win_length//4))
        ax.set_title("Gamma weights")
        cb=plt.colorbar(g,ax=ax)
        cb.set_label("Weight (a.u.)",rotation=270,labelpad=20)
        if save_path != '':
            plt.savefig(os.path.join(save_path,f'gamma_weights_{run}.svg'),transparent=True)
        plt.show()
        plt.close()
    
        ax = plt.gca()
        g=ax.matshow(weights_om,aspect='auto',vmin=-max_max,vmax=max_max,cmap='RdBu_r',vmin=-0.01,vmax=0.01)
        ax.set_ylabel("Neuron number")
        ax.set_xlabel("Lag (ms)")
        ax.set_xticks(range(0,win_length,win_length//4))
        ax.set_xticklabels(range(-win_length,0,win_length//4))#[-20,-15,-10,-5,-1])
        ax.set_title("Omega weights")
        cb=plt.colorbar(g,ax=ax)
        cb.set_label("Weight (a.u.)",rotation=270,labelpad=20)
        if save_path != '':
            plt.savefig(os.path.join(save_path,f'omega_weights_{run}.svg'),transparent=True)
        plt.show()
        plt.close()

    
        choices = np.random.choice(len(spike_designs),n_vocs_vis,replace=False)
    
    
        for jj,c in enumerate(choices):
        
            gamma,omega,spikes = gammas[c],omegas[c],spike_designs[c]
        
            pred_gamma,pred_omega = gam_lr.predict(spikes.reshape(spikes.shape[0],-1)),om_lr.predict(spikes.reshape(spikes.shape[0],-1))
            pred_gamma_null,pred_omega_null = gam_lr_hardnull.predict(spikes.reshape(spikes.shape[0],-1)),om_lr_hardnull.predict(spikes.reshape(spikes.shape[0],-1))
        
            t = np.arange(0,len(omega)*0.001 + 0.0005,0.001)[:len(omega)]*1000
            fig,axs = plt.subplots(nrows=2,ncols=1,figsize=(10,10))
            axs[0].plot(t,omega,label='ouroboros')
            axs[0].plot(t,pred_omega,label='predicted')
            axs[0].plot(t,pred_omega_null,label='null model')
            axs[1].plot(t,gamma,label='ouroboros')
            axs[1].plot(t,pred_gamma,label='predicted')
            axs[1].plot(t,pred_gamma_null,label='null model')

            axs[0].set_xticks([])
            axs[1].set_xlabel("Time (ms)")
            axs[0].set_ylabel(r"$\omega$")
            axs[1].set_ylabel(r"$\gamma$")
            axs[0].legend()
            for ax in axs:
                ax.spines[['top','right']].set_visible(False)
            plt.savefig(os.path.join(save_path,f'model_{run}_recon_{jj}.svg'),transparent=True)
            plt.show()
            plt.close()


    weight_mats_omega = np.stack(weight_mats_omega,axis=0)
    weight_mats_gamma = np.stack(weight_mats_gamma,axis=0)

    plt.rcParams['svg.fonttype'] = 'none'

    mu_omega,mu_gamma = np.nanmean(weight_mats_omega,axis=0),np.nanmean(weight_mats_gamma,axis=0)
    max_gam = np.amax(np.abs(mu_gamma))
    max_om = np.amax(np.abs(mu_omega))
    max_max = max(max_gam,max_om)    
    fig,(ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(15,7),width_ratios=(7,7,1))
    g=ax1.matshow(weights_gam,aspect='auto',vmin=-max_max,vmax=max_max,cmap='RdBu_r',origin='lower',vmin=-0.01,vmax=0.01)
    ax1.set_ylabel("Neuron number")
    ax1.set_xlabel("Lag (ms)")
    ax1.set_xticks(range(0,win_length,win_length//4))
    ax1.set_xticklabels(range(-win_length,0,win_length//4))
    ax1.set_title("Gamma weights")
    #cb=plt.colorbar(g,ax=ax1)
    #cb.set_label("Weight (a.u.)",rotation=270,labelpad=20)
    
    g=ax2.matshow(weights_om,aspect='auto',vmin=-max_max,vmax=max_max,cmap='RdBu_r',origin='lower',vmin=-0.01,vmax=0.01)
    ax2.set_ylabel("Neuron number")
    ax2.set_xlabel("Lag (ms)")
    ax2.set_xticks(range(0,win_length,win_length//4))
    ax2.set_xticklabels(range(-win_length,0,win_length//4))#[-20,-15,-10,-5,-1])
    ax2.set_title("Omega weights")
    cb=plt.colorbar(g,cax=ax3)
    cb.set_label("Weight (a.u.)",rotation=270,labelpad=20)
    plt.tight_layout()
    if save_path != '':
        plt.savefig(os.path.join(save_path,f'average_omega_gamma_weights.svg'),transparent=True)
    plt.show()
    plt.close()

        
    return r2s_omega,r2s_gamma,r2s_omega_null,r2s_gamma_null,r2s_omega_hardnull,r2s_gamma_hardnull