"""
This script sets up an afferent inhomogenous Poisson process onto the populations
"""

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import brian2, string
import numpy as np
from ntwk_stim.poisson_generator import spikes_from_time_varying_rate,\
    deal_with_multiple_spikes_per_bin

def build_aff_to_pop_matrix(afferent_pop, target_pop, Model,
                            SEED=3):
    """
    Generates the connectivity matrix for a random projection from 
    one population to the other !
    """
    np.random.seed(SEED) # insure a precise seed !
    
    N_source_pop = Model['N_'+afferent_pop]
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
    print(additional_spikes)

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
        
        print(additional_spikes)
        
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



# def set_spikes_from_time_varying_rate_synchronous(time_array, rate_array,
#                                                   DUPLICATION_MATRIX, AFF_TO_POP_MATRIX,
#                                                   with_time_shift_synchronous_input=0.,
#                                                   SEED=1):
#     """
#     here, we don't assume that all inputs are decorrelated, we actually
#     model a population of N neurons and just produce spikes according
#     to the "rate_array" frequency
#     """
#     ## time_array in ms !!
#     # so multplying rate array

#     np.random.seed(SEED+18) # setting the seed !
        
#     N_independent = DUPLICATION_MATRIX.shape[0] # 
    
#     pre_indices, pre_times = [], []
#     DT = (time_array[1]-time_array[0])
    
#     # trivial way to generate inhomogeneous poisson events
#     for it in range(len(time_array)):
#         rdm_num = np.random.random(N_independent)
#         for ii in np.arange(N_independent)[rdm_num<DT*rate_array[it]*1e-3/N_independent]: # need to divide by duplicated events
#             for jj in DUPLICATION_MATRIX[ii, :]:
#                 pre_indices.append(jj)
#                 # true time with a time shift (0 by default)
#                 pre_times.append(time_array[it]+with_time_shift_synchronous_input*np.random.randn())
            
#     indices, times = np.empty(0, dtype=np.int), np.empty(0, dtype=np.float64)
#     for ii, tt in zip(pre_indices, pre_times):
#         indices = np.concatenate([indices, np.array(AFF_TO_POP_MATRIX[ii,:], dtype=int)]) # all the indices
#         times = np.concatenate([times, np.array([tt for j in range(len(AFF_TO_POP_MATRIX[ii,:]))])])

        
#     # because brian2 can not handle multiple spikes in one bin, we shift them by dt when concomitant
#     success, nn = False, 0
#     while (not success) and (nn<4):
#         indices, times, success = deal_with_multiple_spikes_per_bin(indices, times, DT)
#         nn+=1
                    
#     return indices, times*brian2.ms, np.array(pre_indices), np.array(pre_times)



# def construct_feedforward_input_synchronous(NTWK,
#                                             target_pop,
#                                             afferent_pop,
#                                             N_source, N_target, N_duplicate,
#                                             t, rate_array,
#                                             with_presynaptic_spikes=False,
#                                             with_time_shift_synchronous_input=0.,
#                                             with_neuron_shift_synchronous_input=0,
#                                             with_neuronpop_shift_synchronous_input=False,
#                                             AFF_TO_POP_MATRIX=None,
#                                             SEED=1):
#     """
#     POPS and AFFERENCE_ARRAY should be 1D arrrays as their is only one 
#     source population

#     'pop_for_conductance' is the string identifying the source conductance
#     that will be incremented by the afferent input !!
#     """

#     np.random.seed(SEED) # setting the seed !

#     Model = NTWK['Model']
    
#     N_independent = int(N_source/N_duplicate)
#     N_source = N_independent*N_duplicate # N_source needs to be a multiple of N_duplicate

#     DUPLICATION_MATRIX = np.array([\
#                               np.random.choice(np.arange(N_source), N_duplicate, replace=False)\
#                               for k in range(N_independent)])
    
#     for ii in range(with_neuron_shift_synchronous_input):
#         for k in range(N_independent):
#             DUPLICATION_MATRIX[k][ii] = np.random.randint(N_source)
    
#     Nsyn = int(Model['p_'+afferent_pop+'_'+target_pop]*N_target)

#     AFF_TO_POP_MATRIX = aff_to_pop_matrix(N_source, N_target, Nsyn)
    
#     if with_neuronpop_shift_synchronous_input:
#         # we shift the stimulus above the N_target neurons
#         AFF_TO_POP_MATRIX += N_target
    
#     indices, times, pre_indices, pre_times = set_spikes_from_time_varying_rate_synchronous(\
#                                                 t, rate_array,\
#                                                 DUPLICATION_MATRIX, AFF_TO_POP_MATRIX,\
#                                                 with_time_shift_synchronous_input=with_time_shift_synchronous_input,
#                                                 SEED=(SEED+2)**2%100)
#     #finding the target pop in the brian2 objects
#     ipop = np.argwhere(NTWK['POPULATIONS']==target_pop).flatten()[0]
#     Qsyn = Model['Q_'+afferent_pop+'_'+target_pop]
    
#     spikes = brian2.SpikeGeneratorGroup(NTWK['POPS'][ipop].N, indices, times, sorted=False)
#     pre_increment = 'G'+afferent_pop+target_pop+' += w'
#     synapse = brian2.Synapses(spikes, NTWK['POPS'][ipop], on_pre=pre_increment,\
#                                     model='w:siemens')
#     synapse.connect('i==j')
#     synapse.w = Qsyn*brian2.nS
    
#     NTWK['PRE_SPIKES'].append(spikes)
#     NTWK['PRE_SYNAPSES'].append(synapse)
    
#     if with_presynaptic_spikes:
#         if 'iRASTER_PRE' in NTWK.keys():
#             NTWK['iRASTER_PRE'].append(indices)
#             NTWK['tRASTER_PRE'].append(times)
#         else: # we create the key
#             NTWK['iRASTER_PRE'] = [indices]
#             NTWK['tRASTER_PRE'] = [times]

#         if 'iRASTER_PRE_in_terms_of_Pre_Pop' in NTWK.keys():
#             NTWK['iRASTER_PRE_in_terms_of_Pre_Pop'].append(pre_indices)
#             NTWK['tRASTER_PRE_in_terms_of_Pre_Pop'].append(pre_times)
#         else: # we create the key
#             NTWK['iRASTER_PRE_in_terms_of_Pre_Pop'] = [pre_indices]
#             NTWK['tRASTER_PRE_in_terms_of_Pre_Pop'] = [pre_times]
            
#     return AFF_TO_POP_MATRIX

            
if __name__=='__main__':

    print('test')
    print(aff_to_pop_matrix(3, 10, 2))

