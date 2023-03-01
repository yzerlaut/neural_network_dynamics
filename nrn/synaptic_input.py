import sys, pathlib, os
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import ntwk # my custom layer on top of Brian2

def spread_synapses_on_morpho(SEGMENTS, density,
                              cond = None,
                              density_factor=1.,
                              with_randomness=True,
                              verbose=True):
    """
    put "density_factor=1./100./1e-12" if you want to express in units synapses/(100um2)

    """
    nseg = len(SEGMENTS['x'])
    if cond is None:
        cond = np.ones(nseg, dtype=bool) # True by default

    Ntot_synapses = int(np.sum(SEGMENTS['area'][cond])*density*density_factor)
    if verbose:
        print('Spreading', Ntot_synapses, 'synapses over the segments')
    
    pre_index_to_segment = np.zeros(Ntot_synapses)
    N_synapses_per_segment = np.zeros(nseg)

    i_pre_index = 0
    for i in np.arange(nseg)[cond]:
        area = SEGMENTS['area'][i]
        N_synapses = int(area*density*density_factor)
        N_synapses_per_segment[i] = N_synapses
        # for each pre index, we give it a segment location
        pre_index_to_segment[i_pre_index:i_pre_index+N_synapses] = i 
        i_pre_index += N_synapses

    i=0
    while (i_pre_index<Ntot_synapses) and i<10000:
        new_cond = (N_synapses_per_segment==0) & cond
        if with_randomness:
            synapse_weight = np.random.uniform(np.sum(new_cond))*SEGMENTS['area'][new_cond]
        else:
            synapse_weight = SEGMENTS['area'][new_cond]
        i0 = np.arange(nseg)[new_cond][np.argmax(synapse_weight)]
        N_synapses_per_segment[i0] = 1
        pre_index_to_segment[i_pre_index] = i0
        i_pre_index += 1
        i+=1
        
    if i==10000:
        print('/!\ Pb with the spread of synapses !! /!\ ')
        
    return Ntot_synapses, np.array(pre_index_to_segment, dtype=int), N_synapses_per_segment


def process_and_connect_event_stimulation(neuron, spike_IDs, spike_times,
                                          pre_index_to_segment,
                                          SYNAPSES_EQUATIONS,
                                          ON_EVENT):
    """
    stim = {'ID':[array of event ID in terms of pre],
            'time': [array of time of those events] }
    """

    stimulation = ntwk.SpikeGeneratorGroup(len(pre_index_to_segment),
                                           np.array(spike_IDs, dtype=int),
                                           spike_times*ntwk.ms)
    
    ES = ntwk.Synapses(stimulation, neuron,
                       model=SYNAPSES_EQUATIONS,
                       on_pre=ON_EVENT, method='exponential_euler')

    for ipre, iseg_post in enumerate(pre_index_to_segment):
        ES.connect(i=ipre, j=iseg_post)

    return stimulation, ES

