"""
This script sets up an afferent inhomogenous Poisson process onto the populations
"""

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import brian2, string
import numpy as np
from .poisson_generator import spikes_from_time_varying_rate, deal_with_multiple_spikes_per_bin


def translate_aff_spikes_into_syn_target_events(source_ids, source_times,
                                                CONN_MATRIX):
    indices, times = [], []
    for s, t in zip(source_ids, source_times):
        for j in CONN_MATRIX[s,:]:
            indices.append(j)
            times.append(t)
    return np.array(indices, dtype=int), np.array(times)


def construct_feedforward_input(NTWK, target_pop, afferent_pop,\
                                t, rate_array,\
                                additional_spikes={'indices':[], 'times':[]},
                                additional_spikes_in_terms_of_pre_pop={'indices':[], 'times':[]},
                                verbose=False,
                                SEED=1):
    """
    This generates an input asynchronous from post synaptic neurons to post-synaptic neurons

    POPS and AFFERENCE_ARRAY should be 1D arrrays as their is only one 
    source population

    'pop_for_conductance' is the string identifying the source conductance
    that will be incremented by the afferent input !!

    """

    Model = NTWK['Model']
    
    # Synapses  number ?
    Nsyn = Model['p_'+afferent_pop+'_'+target_pop]*Model['N_'+afferent_pop]
    
    if Nsyn>0: # if non-zero projection [...]
        
        # extract parameters of the afferent input
        Qsyn = Model['Q_'+afferent_pop+'_'+target_pop]
        
        #finding the target pop in the brian2 objects
        ipop = np.argwhere(NTWK['POPULATIONS']==target_pop).flatten()[0]
        
        if verbose:
            print('drawing Poisson process for afferent input [...]')
            
        indices, times = spikes_from_time_varying_rate(t, rate_array,\
                                                       NTWK['POPS'][ipop].N,
                                                       Nsyn,
                                                       SEED=(SEED+2)**2%100)

        # adding the additional spikes (1)
        indices = np.concatenate([indices, additional_spikes['indices']])
        times = np.concatenate([times, additional_spikes['times']])
        
        # adding the additional spikes (2)
        if len(additional_spikes_in_terms_of_pre_pop['indices'])>0:
            try:
                Matrix = NTWK['M_conn_'+afferent_pop+'_'+target_pop]
                indices2, times2 = translate_aff_spikes_into_syn_target_events(np.array(additional_spikes_in_terms_of_pre_pop['indices'], dtype=int),
                                                                               additional_spikes_in_terms_of_pre_pop['times'], Matrix)
            except KeyError:
                print("""
                -------------------------------------------------
                Need to construct the afference to use this, with:
                ntwk.build_fixed_afference(NTWK,
                                           ['AffExc'],
                                           ['Exc', 'Inh', 'DsInh'])
                """)
        else:
            indices2, times2 = [], []
                      
        indices = np.concatenate([indices, indices2, additional_spikes['indices']])
        times = np.concatenate([times, times2, additional_spikes['times']])
                
        # insuring no more than one prespike per bin
        indices, times = deal_with_multiple_spikes_per_bin(indices, times, t, verbose=verbose)

        # incorporating into Brian2 objects
        spikes = brian2.SpikeGeneratorGroup(NTWK['POPS'][ipop].N, indices, times*brian2.ms)
        # sorted = True, see "deal_with_multiple_spikes_per_bin"

        if ('psyn_%s_%s' % (afferent_pop, target_pop) in Model):
            psyn = Model['psyn_%s_%s' % (afferent_pop, target_pop)] # probability of release
            on_pre = 'G%s%s_post+=(rand()<%.3f)*w' % (afferent_pop, target_pop, psyn)
        else:
            on_pre = 'G'+afferent_pop+target_pop+' += w'

        synapse = brian2.Synapses(spikes, NTWK['POPS'][ipop], model='w:siemens', on_pre=on_pre)
        synapse.connect('i==j')
        synapse.w = Qsyn*brian2.nS

        NTWK['PRE_SPIKES'].append(spikes)
        NTWK['PRE_SYNAPSES'].append(synapse)
        
    else:
        print('Nsyn = 0 for', afferent_pop+'_'+target_pop)
        spikes, synapse, indices, times, pre_indices, pre_times = None, None, [], [], [], []
    
    # afferent array
    NTWK['Rate_%s_%s' % (afferent_pop, target_pop)] = rate_array
    
    # storing quantities:
    if 'iRASTER_PRE' in NTWK.keys():
        NTWK['iRASTER_PRE'].append(indices)
        NTWK['tRASTER_PRE'].append(times)
    else: # we create the key
        NTWK['iRASTER_PRE'] = [indices]
        NTWK['tRASTER_PRE'] = [times]


    
    # if 'iRASTER_PRE_in_terms_of_Pre_Pop' in NTWK.keys():
    #     NTWK['iRASTER_PRE_in_terms_of_Pre_Pop'].append(pre_indices)
    #     NTWK['tRASTER_PRE_in_terms_of_Pre_Pop'].append(pre_times)
    # else: # we create the key
    #     NTWK['iRASTER_PRE_in_terms_of_Pre_Pop'] = [pre_indices]
    #     NTWK['tRASTER_PRE_in_terms_of_Pre_Pop'] = [pre_times]



        
if __name__=='__main__':

    print('test')
    print(build_aff_to_pop_matrix(3, 10, 2))

