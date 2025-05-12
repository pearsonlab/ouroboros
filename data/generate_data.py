import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax

import diffrax 

import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'

from scipy.signal import stft
import scipy.io as sio
import numpy as np
import pickle

def make_pulse_sequence(pulse_fun, arg_list):
    def applicator(carry, args):
       t = carry
       return t, pulse_fun(t, *args)
    
    def scan_and_sum(t):
        _, out = lax.scan(applicator, t, arg_list)
        return jnp.sum(out)

    return scan_and_sum

###### vS tension function #####
kshape = 5  # shape parameter of gamma function
kscale = 0.025  # rate parameter of gamma function (s)
kpeak = 0.06  # peak value (Volts)
klocs = jnp.array([0, 0.5, 1, 1.5])
def make_tension_pulse_fn(shape=1, scale=1, peak=1):
    norm = peak * jnp.exp(shape - shape * jnp.log(shape) - shape * jnp.log(scale)) 

    fn = lambda t, loc: norm * jnp.exp((t - loc)/scale) * jnp.maximum((loc - t), 0)**shape 

    return fn
####################################

### dTb tension function ###########
dfreq = 2 
#dlocs1 = jnp.arange(0.1, t_axis[-1], 1/dfreq)  # pressure pulse frequency (Hz)
#dlocs2 = jnp.arange(0.45, t_axis[-1], 1/dfreq)  # pressure pulse frequency (Hz)
#dlocs = jnp.sort(jnp.concatenate([dlocs1, dlocs2]))
dA = jnp.array([0.05, 0.02, 0.01, 0.05, 0.03,])
dwid=0.01
def gpulse(t,loc,amp):

    return amp*jnp.exp(-0.5 * (t - loc)**2/dwid**2)
########################################

##### Pressure functions ######
pA = 0.025
p0 = -0.005
pwid=0.08
def ppulse(t,loc):

    return pA * jnp.exp(-0.5 * (t - loc)**2/pwid**2) + p0
####################################

def gradfun(t, y, args):
    # params: eps1, eps2, beta1, beta2, C, delta
    #print(y.shape)
    params, extra_args = args
    K, D, P = extra_args
    t_arr = jnp.array([t])
    eps = (params[0] + params[1] * K(t_arr)) * 1e8
    B = (params[2] + params[3] * P(t_arr)) * 1e3
    C = params[4] * 1e8
    D0 = params[5] * D(t_arr) * 1e7

    xdot = y[1] 
    ydot = -eps * y[0] - C * y[0]**2 * y[1] + B * y[1] - D0

    return jnp.array((xdot, ydot[0]))

SR = 4e4
DFREQ=2

kshape,kscale,kpeak = 5,0.025,0.06
eps1 = 1.25e8
eps2 = 7.5e9
beta1 = -2e3
beta2 = 5.3e5  # NOTE: 10x higher than in paper!
C = 2e8
delta = 15e6
params_true = jnp.array([eps1/1e8, eps2/1e8, beta1/1e3, beta2/1e3, C/1e8, delta/1e7])


def generate_vocalization(key,length=1.2,n_p=4,n_k=6,dA_max=0.1,dA_min=0.01,save=True,saveloc=''):

    t_axis = jnp.arange(0, length, 1/SR)
    
    saveat = diffrax.SaveAt(ts=jnp.linspace(0, length, int(SR)))


    #key = jax.random.key(1234)
    key,pKey,kKey1,kKey2,dKey = jax.random.split(key,5)

    plocs = jax.random.uniform(pKey,maxval=length,shape=(n_p,))
    #plocs = jnp.linspace(0.2,length,n_p)
    #klocs = jax.random.uniform(kKey,maxval=length,shape=(n_k,))
    klocs=plocs - jax.random.uniform(kKey1,maxval=0.1,minval=-0.1,shape=plocs.shape)
    klocs = jax.random.choice(kKey2,klocs,shape=(n_k,),replace=False)
    klocs = jnp.sort(jnp.concatenate([klocs,klocs + 0.3]))
    
    
    dlocs1 = jnp.arange(0.1, t_axis[-1], 1/DFREQ)  # pressure pulse frequency (Hz)
    dlocs2 = jnp.arange(0.45, t_axis[-1], 1/DFREQ)  # pressure pulse frequency (Hz)
    dlocs = jnp.sort(jnp.concatenate([dlocs1, dlocs2]))
    
    dA = jax.random.uniform(dKey,maxval=dA_max,minval=dA_min,shape=dlocs.shape)
    
    P = jax.vmap(make_pulse_sequence(ppulse, (plocs,)))
    pulse = make_tension_pulse_fn(shape=kshape, scale=kscale, peak=kpeak)
    K = jax.vmap(make_pulse_sequence(pulse, (klocs,)))
    D = jax.vmap(make_pulse_sequence(gpulse, (dlocs, dA)))

    print("integrating ODE")
    term = diffrax.ODETerm(gradfun)
    saveat = diffrax.SaveAt(ts=jnp.linspace(0, length, int(SR)))
    solver = diffrax.Dopri5()
    stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-5)
    soln = diffrax.diffeqsolve(term, solver, t0=0, t1=length, dt0=0.5/SR, y0=jnp.array((0, 0)), saveat=saveat,
                  stepsize_controller=stepsize_controller, args=(params_true, (K, D, P)), max_steps=int(1e6))

    #plt.plot(soln.ts, soln.ys[:, 0])
    #plt.show()
    #plt.close()

    
    audio = jnp.interp(t_axis, soln.ts, soln.ys[:, 0])
    kt,dt,pt = K(t_axis),D(t_axis),P(t_axis)

    if save:
        sio.wavfile.write(saveloc + '.wav',int(SR),np.array(audio))
        with open(saveloc + 'inputs.pkl','wb') as f:
            pickle.dump([kt,dt,pt],f)



    return t_axis,audio