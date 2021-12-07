import numpy as np
import numba
import time, sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from cells.cell_library import get_neuron_params, change_units_to_SI

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
    spike_events = np.concatenate([spike_events, [t[-1]+1.]]) # adding a final spike for the while loop
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
        Ai, Vi, Ti = 0., Vthre, 1e3

    return El, Gl, Cm, Vthre, Vreset, vspike, vpeak,\
        Trefrac, delta_v, a, b, tauw, Vi, Ti, Ai
                     
def iAdExp_sim(t, G_ARRAY, E_ARRAY, I,
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
    V = Vreset*np.ones(len(t), dtype=np.float64)
    spikes = []
    theta=Vthre*np.ones(len(t), dtype=np.float64) # initial adaptative threshold value
    dt = t[1]-t[0]

    w, i_exp = 0., 0. # w and i_exp are the exponential and adaptation currents

    for i in range(len(t)-1):
        
        # adaptation current
        w = w + dt/tauw*(a*(V[i]-El)-w)
        
        # spiking no-linearity
        i_exp = Gl*delta_v*np.exp((V[i]-Vthre)*one_over_delta_v)
        
        # synaptic currents
        Isyn = 0
        for g in range(len(E_ARRAY)):
            Isyn += G_ARRAY[g,i]*(E_ARRAY[g]-V[i])
            
        ## Vm dynamics calculus
        if (t[i]-last_spike)>Trefrac: # only when non refractory
            V[i+1] = V[i] + dt/Cm*(I[i] + i_exp - w +\
                                Gl*(El-V[i]) + Isyn )
        # then threshold
        theta_inf_v = Vthre + Ai*0.5*(1+np.sign(V[i]-Vi))*(V[i]-Vi)
        theta[i+1] = theta[i] + dt/Ti*(theta_inf_v - theta[i])

        # threshold mechanism with one step at Vpeak
        # if V[i]==vpeak:
        #     V[i+1] = Vreset
        if V[i+1] >= theta[i+1]+5.*delta_v:
            V[i+1] = Vreset
            w = w + b # then we increase the adaptation current
            last_spike = t[i+1]
            spikes.append(t[i+1])

    return V, theta, spikes

@numba.jit(nopython=True)
def iAdExp_sim_fast(t, G_ARRAY, E_ARRAY, I,
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
    V1, V0 = Vreset, Vreset
    nspikes = 0 # just counting spikes
    theta0, theta1 = Vthre, Vthre
    dt = t[1]-t[0]

    w, i_exp = 0., 0. # w and i_exp are the exponential and adaptation currents

    for i in range(len(t)-1):
        V0 = V1
        # adaptation current
        w = w + dt/tauw*(a*(V0-El)-w)
        
        # spiking no-linearity
        i_exp = Gl*delta_v*np.exp((V0-Vthre)*one_over_delta_v)
        
        # synaptic currents
        Isyn = 0
        for g in range(len(E_ARRAY)):
            Isyn += G_ARRAY[g,i]*(E_ARRAY[g]-V0)
            
        ## Vm dynamics calculus
        if (t[i]-last_spike)>Trefrac: # only when non refractory
            V1 = V0 + dt/Cm*(I[i] + i_exp - w +\
                                Gl*(El-V0) + Isyn )
        # then threshold
        theta_inf_v = Vthre + Ai*0.5*(1+np.sign(V0-Vi))*(V0-Vi)
        theta1 = theta0 + dt/Ti*(theta_inf_v - theta0)

        if V1 >= theta1+5.*delta_v:
            V1 = Vreset
            w = w + b # then we increase the adaptation current
            last_spike = t[i+1]
            nspikes +=1

    return nspikes/t[-1] # return only the firing rate

####################################################################
############ One simulation ########################################
####################################################################

def single_experiment(params, SYN_POPS, RATES, seed=3,\
                      return_threshold=False, firing_rate_only=False, dt=0.1*1e-3, tstop=500.*1e-3):

    t = np.arange(int(tstop/dt))*dt
    I = 0*t # no current input
    G_ARRAY, E_ARRAY = [], []
    for syn, rate in zip(SYN_POPS, RATES):

        G_ARRAY.append(generate_conductance_shotnoise(rate, t, syn['N']*syn['pconn'],\
                                                      syn['Q'], syn['Tsyn'], g0=0,
                                                      seed=(seed+3)*(int(rate*1e3%18))))
        E_ARRAY.append(syn['Erev'])
        
    G_ARRAY, E_ARRAY = np.array(G_ARRAY), np.array(E_ARRAY)
    

    if firing_rate_only:
        return iAdExp_sim_fast(t, G_ARRAY, E_ARRAY, I, *pseq_iAdExp(params))
    else:
        v, theta, spikes = iAdExp_sim(t, G_ARRAY, E_ARRAY, I, *pseq_iAdExp(params))
        if return_threshold:
            return t, v, theta, spikes
        else:
            return t, v, spikes 

if __name__=='__main__':

    ### ========================================================================= ###
    ## Checking that this custom simulation scheme gives the same output than Brian 2 
    ### ========================================================================= ###
    
    dt, tstop = 0.05, 10000.
    params = {'N':1,
              'Gl':10., 'Cm':200.,'Trefrac':5.,
              'El':-70., 'Vthre':-50., 'Vreset':-70.,
              'a': 0.0, 'tauw': 1e9, 'b': 0.0, 'delta_v':0}

    SYN_POPS = [{'name':'exc1', 'Erev': 0.0, 'N': 2000, 'Q': 1., 'Tsyn': 3., 'pconn': 0.1},
                {'name':'exc2', 'Erev': 0.0, 'N': 2000, 'Q': 8., 'Tsyn': 3., 'pconn': 0.1},
                {'name':'inh1', 'Erev': -80.0, 'N': 500, 'Q': 10., 'Tsyn':3., 'pconn': 0.1},
                {'name':'inh2', 'Erev': -80.0, 'N': 500, 'Q': 20., 'Tsyn': 3., 'pconn': 0.1}]
    RATES = [.5,.3,.9,.2]

    # BRIAN2 simulation
    from transfer_functions.single_cell_protocol import run_sim as single_cell_sim
    data = single_cell_sim(params, SYN_POPS, RATES,
                       tstop=tstop, dt=dt, with_Vm=1, SEED=8)

    ## CHANGING TO SI UNITS ##
    dt, tstop = 1e-3*dt, 1e-3*tstop
    change_units_to_SI(params) # neuronal params
    for syn in SYN_POPS:
        change_units_to_SI(syn) # synaptic params

    t, v, spikes = single_experiment(params, SYN_POPS, RATES, dt=dt, tstop=tstop)

    import matplotlib.pylab as plt
    plt.hist(data['Vm'][0])
    plt.hist(1e3*v, alpha=.5)
    plt.show()
