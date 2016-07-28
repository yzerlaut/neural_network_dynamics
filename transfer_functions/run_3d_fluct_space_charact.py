import numpy as np
import time
import sys
sys.path.append('../')
from cells.cell_library import get_neuron_params

####################################################################
############ Functions for the spiking dynamics ###########
####################################################################

def generate_conductance_shotnoise(freq, t, N, Q, Tsyn, g0=0, seed=0):
    """
    generates a shotnoise convoluted with a waveform
    frequency of the shotnoise is freq,
    K is the number of synapses that multiplies freq
    g0 is the starting value of the shotnoise
    """
    if freq==0:
        # print "problem, 0 frequency !!! ---> freq=1e-9 !!"
        freq=1e-9
    upper_number_of_events = max([int(3*freq*t[-1]*N),1]) # at least 1 event
    np.random.seed(seed=seed)
    spike_events = np.cumsum(np.random.exponential(1./(N*freq),\
                             upper_number_of_events))
    g = np.ones(t.size)*g0 # init to first value
    dt, t = t[1]-t[0], t-t[0] # we need to have t starting at 0
    # stupid implementation of a shotnoise
    event = 0 # index for the spiking events
    for i in range(1,t.size):
        g[i] = g[i-1]*np.exp(-dt/Tsyn)
        while spike_events[event]<=t[i]:
            g[i]+=Q
            event+=1
    return g

### ================================================
### ======== iAdExp model (general) ================
### == extension of LIF, iLIF, EIF, AdExp, ...
### ================================================

def pseq_iAdExp(cell_params):

    El, Gl = cell_params['El'], cell_params['Gl']
    Cm = cell_params['Cm']
    
    Vthre, Vreset = cell_params['Vthre'], cell_params['Vreset']

    # adaptation variables
    a, b, tauw = cell_params['a'],\
                     cell_params['b'], cell_params['tauw']

    # spike variables
    Trefrac, delta_v = cell_params['Trefrac'], cell_params['delta_v']

    vspike = Vthre+5.*delta_v
    vpeak = 0
    
    # inactivation variables
    if 'Ai' in cell_params.keys():
        Vi, Ti = cell_params['Vi'], cell_params['Ti']
        Ai = cell_params['Ai']
    else:
        Ai, Vi, Ti = 0., -50e-3, 1e3

    return El, Gl, Cm, Vthre, Vreset, vspike, vpeak,\
                     Trefrac, delta_v, a, b, tauw, Vi, Ti, Ai
                     
# @numba.jit('u1[:](f8[:], f8[:], f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8)')
def iAdExp_sim(t, I, Gs, muV,
         El, Gl, Cm, Vthre, Vreset, vspike, vpeak,\
         Trefrac, delta_v, a, b, tauw, Vi, Ti, Ai):
    """ functions that solve the membrane equations for the
    adexp model for 2 time varying excitatory and inhibitory
    conductances as well as a current input
    returns : v, spikes
    """

    if delta_v==0: # i.e. Integrate and Fire
        one_over_delta_v = 0
    else:
        one_over_delta_v = 1./delta_v
        
    vspike=Vthre+5.*delta_v # practical threshold detection
            
    last_spike = -np.inf # time of the last spike, for the refractory period
    V, spikes = Vreset*np.ones(len(t), dtype=np.float), []
    theta=Vthre*np.ones(len(t), dtype=np.float) # initial adaptative threshold value
    dt = t[1]-t[0]

    w, i_exp = 0., 0. # w and i_exp are the exponential and adaptation currents

    for i in range(len(t)-1):
        w = w + dt/tauw*(a*(V[i]-El)-w) # adaptation current
        i_exp = Gl*delta_v*np.exp((V[i]-Vthre)*one_over_delta_v) 
        
        if (t[i]-last_spike)>Trefrac: # only when non refractory
            ## Vm dynamics calculus
            V[i+1] = V[i] + dt/Cm*(I[i] + i_exp - w +\
                                Gl*(El-V[i]) + Gs*(muV-V[i]) )

        # then threshold
        theta_inf_v = Vthre + Ai*0.5*(1+np.sign(V[i]-Vi))*(V[i]-Vi)
        theta[i+1] = theta[i] + dt/Ti*(theta_inf_v - theta[i])
        
        if V[i+1] >= theta[i+1]+5.*delta_v:
            
            V[i+1] = Vreset # non estethic version
            
            w = w + b # then we increase the adaptation current
            last_spike = t[i+1]
            spikes.append(t[i+1])

    return V, theta, np.array(spikes)


####################################################################
####### Calculating the input need to produce given fluct. ########
####################################################################

def params_variations_calc(muGn, muV, sV, Ts_ratio, params):
    """
    input should be numpy arrays !!

    We solve the equations:
    Ts = Tv - \frac{C_m}{\mu_G}
    Q \, T_S (\nu_e + \nu_i) = \mu_G
    Q \, T_S (\nu_e E_e + \nu_i E_i) = \mu_G \mu_V - g_L E_L
    Q^2 \, T_S^2 \, big( \nu_e (E_e-\mu_V)^2 +
        \nu_i (E_i - \mu_V)^2 \big) = 2 \mu_G^2 \tau_V \sigma_V^2

    return numpy arrays !!
    """

    Gl, Cm, El = params['Gl'], params['Cm'], params['El']
    Tm0 = Cm/Gl
    Ts = Ts_ratio*Tm0
    DV = params['Driving_Force']
    muG = muGn*Gl
    Gs = muG-Gl # shunt conductance
    Tv = Ts+Tm0/muGn
    I0 = Gl*(muV-El) # current to bring at mean !
    f = 2000.+0*I0 #Hz
    Q = muG*sV*np.sqrt(Tv/f)/Ts/DV
    
    return I0, Gs, f, Q, Ts

####################################################################
############ One simulation ########################################
####################################################################

def single_experiment(t, I0, Gs, f, Q, Ts, muV,\
                      params, MODEL='SUBTHRE', seed=0,\
                      return_threshold=False):

    params = params.copy()
    
    Ge = generate_conductance_shotnoise(f, t, 1.,\
                            Q, Ts, g0=0, seed=seed)
    Gi = generate_conductance_shotnoise(f, t, 1.,\
                            Q, Ts, g0=0, seed=seed**2+1)

    I = np.ones(len(t))*I0+(Ge-Gi)*params['Driving_Force']
    
    v, theta, spikes = iAdExp_sim(t, I, Gs, muV, *pseq_iAdExp(params))

    if return_threshold:
        return v, theta, spikes
    else:
        return v, spikes
        
def make_simulation_for_model(MODEL, args, return_output=False,\
                              sampling='low'):

    if sampling is 'low':
        # discretization and seed
        SEED = np.arange(2)+1
        dt, tstop = 1e-4, 2.
    else:
        SEED = np.arange(3)+1
        dt, tstop = 1e-5, 10.

    params = get_neuron_params(MODEL, SI_units=True)
        
    ### PARAMETERS OF THE EXPERIMENT
    params['RANGE_FOR_3D'] = args.RANGE_FOR_3D
        
    muV_min, muV_max,\
        sV_min1, sV_max1, sV_min2, sV_max2,\
        Ts_ratio = params['RANGE_FOR_3D']
    muGn_min, muGn_max = 1.15, 8.

    ### cell and synaptic parameters, see models.py !!
    sim_params = {'dt':dt, 'tstop':tstop}
    t_long = np.arange(0,int(tstop/dt))*dt

    muV = np.linspace(muV_min, muV_max, args.DISCRET_muV, endpoint=True)

    # trying with the linear autocorrelation 
    args.DISCRET_muG = args.DISCRET_TvN
    Tv_ratio = np.linspace(1./muGn_max+Ts_ratio, 1./muGn_min+Ts_ratio, args.DISCRET_muG, endpoint=True)
    muGn = 1./(Tv_ratio-Ts_ratio)

    muV, sV, muGn = np.meshgrid(muV, np.zeros(args.DISCRET_sV), muGn)
    Tv_ratio = Ts_ratio+1./muGn

    for i in range(args.DISCRET_muV):
        sv1 = sV_min1+i*(sV_min2-sV_min1)/(args.DISCRET_muV-1)
        sv2 = sV_max1+i*(sV_max2-sV_max1)/(args.DISCRET_muV-1)
        for l in range(args.DISCRET_muG):
            sV[:,i,l] = np.linspace(sv1,sv2,args.DISCRET_sV,endpoint=True)

    params['Driving_Force'] = args.Driving_Force
    I0, Gs, f, Q, Ts = params_variations_calc(muGn,muV,sV,\
                                              Ts_ratio*np.ones(muGn.shape),params)

    Fout = np.zeros((args.DISCRET_sV, args.DISCRET_muV, args.DISCRET_muG, len(SEED)))

    for i_muV in range(args.DISCRET_muV):
        print('[[[[]]]]=====> muV : ', round(1e3*muV[0, i_muV, 0],1), 'mV')
        for i_sV in range(args.DISCRET_sV):
            print('[[[]]]====> sV : ', round(1e3*sV[i_sV, i_muV, 0],1), 'mV')
            for ig in range(args.DISCRET_muG):
                print('[]=> muGn : ', round(muGn[i_sV, i_muV, ig],1),\
                    'TvN : ', round(100*Tv_ratio[i_sV, i_muV, ig],1), '%')
                for i_s in range(len(SEED)):
                    v, spikes = single_experiment(\
                        t_long, I0[i_sV, i_muV, ig],\
                        Gs[i_sV, i_muV, ig],\
                        f[i_sV, i_muV, ig],\
                        Q[i_sV, i_muV, ig],\
                        Ts[i_sV, i_muV, ig],\
                        muV[i_sV, i_muV, ig],\
                        params, MODEL=MODEL,
                        seed=SEED[i_s]+i_muV+i_sV+ig)
                    print(spikes, MODEL)

                    Fout[i_sV, i_muV, ig, i_s] =\
                      len(spikes)/t_long.max()

    data_path = 'data/'+MODEL+'.npz'


    D = dict(muV=1e3*muV.flatten(), sV=1e3*sV.flatten(),\
             TvN=Ts_ratio+1./muGn.flatten(),\
             muGn=muGn.flatten(),\
             Fout=Fout.mean(axis=-1).flatten(),\
             s_Fout=Fout.std(axis=-1).flatten(),\
             MODEL=MODEL,\
             Gl=params['Gl'], Cm=params['Cm'], El=params['El'])

    np.savez(data_path,**D)    

if __name__=='__main__':
    # for spiking properties, what model ?? see models.py
    import argparse
    parser=argparse.ArgumentParser(description=
     """ 
     Stimulate a reconstructed cell with a shotnoise and study Vm dynamics
     """
    ,formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument("MODEL", help="Choose a model of NEURON")
    parser.add_argument("--sampling", default='low',\
                        help="turn to 'high' for simulations as in the paper")
    parser.add_argument("--DISCRET_muV", default=4, type=int,\
                        help="discretization of the 3d grid for muV")
    parser.add_argument("--DISCRET_sV", default=8, type=int,\
                        help="discretization of the 3d grid for sV")
    parser.add_argument("--DISCRET_TvN", default=4, type=int,\
                        help="discretization of the 3d grid for TvN")
    parser.add_argument("--Driving_Force", default=20e-3, type=float,\
                        help="static driving force")
    parser.add_argument("--RANGE_FOR_3D", type=float,\
                default=[-70e-3, -55e-3, 5e-3, 9e-3, 1e-3, 5e-3, .25], nargs='+',\
                        help="possibility to explicitely set the 3D range scanned")
    args = parser.parse_args()

    make_simulation_for_model(args.MODEL, args, sampling=args.sampling)
        
