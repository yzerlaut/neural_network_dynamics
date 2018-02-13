"""
This script sets up an afferent inhomogenous Poisson process onto the populations
"""

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import brian2, string
import numpy as np
from ntwk_stim.poisson_generator import spikes_from_time_varying_rate,\
    deal_with_multiple_spikes_per_bin

def build_aff_to_pop_matrix(afferent_pop, target_pop,
                            Model,
                            N_source_pop=None,
                            N_target_pop=None,
                            SEED=3):
    """
    Generates the connectivity matrix for a random projection from 
    one population to the other !

    possibility to subsample the target or source pop through the "N_source_pop" and "N_target_pop" args
    """
    np.random.seed(SEED) # insure a precise seed !

    if N_source_pop is None:
        N_source_pop = Model['N_'+afferent_pop]
    if N_target_pop is None:
        N_target_pop = Model['N_'+target_pop]
    Nsyn_onto_target_from_source = int(Model['p_'+afferent_pop+'_'+target_pop]*Model['N_'+afferent_pop])
    
    return np.array([\
      np.random.choice(np.arange(N_target_pop), Nsyn_onto_target_from_source, replace=False)\
                     for k in range(N_source_pop)], dtype=int)


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
                                AFF_TO_POP_MATRIX=None,
                                additional_spikes={'indices':[], 'times':[]},
                                verbose=False,
                                SEED=1):
    """
    This generates an input asynchronous from post synaptic neurons to post-synaptic neurons

    POPS and AFFERENCE_ARRAY should be 1D arrrays as their is only one 
    source population

    'pop_for_conductance' is the string identifying the source conductance
    that will be incremented by the afferent input !!

    if AFF_TO_POP_MATRIX then fixed pre-pop number: see "poisson_generator.py"
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
            
        indices, times, pre_indices,\
            pre_times = spikes_from_time_varying_rate(\
                                                       t, rate_array,\
                                                       NTWK['POPS'][ipop].N,
                                                       Nsyn,
                                                       AFF_TO_POP_MATRIX=AFF_TO_POP_MATRIX,
                                                       SEED=(SEED+2)**2%100)
        
        # adding the additional spikes
        indices = np.concatenate([indices, additional_spikes['indices']])
        times = np.concatenate([times, additional_spikes['times']])
        
        
        # insuring no more than one prespike per bin
        indices, times = deal_with_multiple_spikes_per_bin(indices, times, t, verbose=verbose)

        # incorporating into Brian2 objects
        spikes = brian2.SpikeGeneratorGroup(NTWK['POPS'][ipop].N, indices, times*brian2.ms)
        pre_increment = 'G'+afferent_pop+target_pop+' += w'
        synapse = brian2.Synapses(spikes, NTWK['POPS'][ipop], on_pre=pre_increment,\
                                        model='w:siemens')
        synapse.connect('i==j')
        synapse.w = Qsyn*brian2.nS

        NTWK['PRE_SPIKES'].append(spikes)
        NTWK['PRE_SYNAPSES'].append(synapse)
        
    else:
        print('Nsyn = 0 for', afferent_pop+'_'+target_pop)
        spikes, synapse, indices, times = None, None, [], []
    
    # storing quantities:
    if 'iRASTER_PRE' in NTWK.keys():
        NTWK['iRASTER_PRE'].append(indices)
        NTWK['tRASTER_PRE'].append(times)
    else: # we create the key
        NTWK['iRASTER_PRE'] = [indices]
        NTWK['tRASTER_PRE'] = [times]
        
    if 'iRASTER_PRE_in_terms_of_Pre_Pop' in NTWK.keys():
        NTWK['iRASTER_PRE_in_terms_of_Pre_Pop'].append(pre_indices)
        NTWK['tRASTER_PRE_in_terms_of_Pre_Pop'].append(pre_times)
    else: # we create the key
        NTWK['iRASTER_PRE_in_terms_of_Pre_Pop'] = [pre_indices]
        NTWK['tRASTER_PRE_in_terms_of_Pre_Pop'] = [pre_times]

if __name__=='__main__':

    print('test')
    print(aff_to_pop_matrix(3, 10, 2))

