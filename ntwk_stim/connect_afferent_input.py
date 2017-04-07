"""
This script sets up an afferent inhomogenous Poisson process onto the populations
"""
import brian2, string
import numpy as np

def set_spikes_from_time_varying_rate(time_array, rate_array, N, Nsyn, SEED=1):
    
    brian2.seed(SEED) # setting the seed !
    
    ## time_array in ms !!
    # so multplying rate array
    
    indices, times = [], []
    DT = (time_array[1]-time_array[0])
    
    # trivial way to generate inhomogeneous poisson events
    for it in range(len(time_array)):
        rdm_num = np.random.random(N)
        for ii in np.arange(N)[rdm_num<DT*Nsyn*rate_array[it]*1e-3]:
            indices.append(ii) # all the indices
            times.append(time_array[it]) # all the same time !

    return np.array(indices), np.array(times)*brian2.ms


def construct_feedforward_input(NTWK, target_pop,
                                afferent_pop,\
                                t, rate_array,\
                                conductanceID='AA',\
                                with_presynaptic_spikes=False,
                                SEED=1):
    """
    POPS and AFFERENCE_ARRAY should be 1D arrrays as their is only one 
    source population

    'pop_for_conductance' is the string identifying the source conductance
    that will be incremented by the afferent input !!
    """

    # number of synapses per neuron
    Nsyn = afferent_pop['pconn']*afferent_pop['N']
    if Nsyn>0:
        indices, times = set_spikes_from_time_varying_rate(\
                            t, rate_array,\
                            target_pop.N, Nsyn, SEED=(SEED+2)**2%100)
        spikes = brian2.SpikeGeneratorGroup(target_pop.N, indices, times)
        pre_increment = 'G'+conductanceID+' += w'
        synapse = brian2.Synapses(spikes, target_pop, on_pre=pre_increment,\
                                        model='w:siemens')
        synapse.connect('i==j')
        synapse.w = afferent_pop['Q']*brian2.nS
    else:
        print('Nsyn = 0')
        spikes, synapse = None, None

    NTWK['PRE_SPIKES'].append(spikes)
    NTWK['PRE_SYNAPSES'].append(synapse)
    
    if with_presynaptic_spikes:
        if 'iRASTER_PRE' in NTWK.keys():
            NTWK['iRASTER_PRE'].append(indices)
            NTWK['tRASTER_PRE'].append(times)
        else: # we create the key
            NTWK['iRASTER_PRE'] = [indices]
            NTWK['tRASTER_PRE'] = [times]

