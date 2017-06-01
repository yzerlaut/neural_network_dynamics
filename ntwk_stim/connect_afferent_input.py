"""
This script sets up an afferent inhomogenous Poisson process onto the populations
"""
import brian2, string
import numpy as np

def set_spikes_from_time_varying_rate(time_array, rate_array, N, Nsyn, SEED=1):
    """

    """
    np.random.seed(SEED) # setting the seed !
    
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


def construct_feedforward_input(NTWK, target_pop, afferent_pop,\
                                t, rate_array,\
                                with_presynaptic_spikes=False,
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
    
    # extract parameters of the afferent input
    Nsyn = Model['p_'+afferent_pop+'_'+target_pop]*Model['N_'+afferent_pop]
    Qsyn = Model['Q_'+afferent_pop+'_'+target_pop]

    #finding the target pop in the brian2 objects
    ipop = np.argwhere(NTWK['POPULATIONS']==target_pop).flatten()[0]
    
    if Nsyn>0:
        if verbose:
            print('drawing Poisson process for afferent input [...]')
        indices, times = set_spikes_from_time_varying_rate(\
                            t, rate_array,\
                            NTWK['POPS'][ipop].N, Nsyn, SEED=(SEED+2)**2%100)
        spikes = brian2.SpikeGeneratorGroup(NTWK['POPS'][ipop].N, indices, times)
        pre_increment = 'G'+afferent_pop+target_pop+' += w'
        synapse = brian2.Synapses(spikes, NTWK['POPS'][ipop], on_pre=pre_increment,\
                                        model='w:siemens')
        synapse.connect('i==j')
        synapse.w = Qsyn*brian2.nS
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


def deal_with_multiple_spikes_within_one_bin(indices, times, DT):
    # nasty loop
    n=0
    for tt in np.unique(times):
        for ii in np.unique(indices):
            i1 = np.argwhere(((times==tt) & (indices==ii))).flatten()
            if len(i1)>1:
                for j, index in enumerate(i1):
                    times[index] += DT*(-1)**j*int(j/2+.5)
                    n+=1
                n-=1 # remove the spike unchanged
    if n==0:
        return indices, times, True
    else:
        print('n=', n, 'spikes were shifted by dt to insure no overlapping presynaptic spikes (Brian2 constraint)')
        return indices, times, False

def set_spikes_from_time_varying_rate_correlated(time_array, rate_array, AFF_TO_POP_MATRIX, SEED=1):
    """
    here, we don't assume that all inputs are decorrelated, we actually
    model a population of N neurons and just produce spikes according
    to the "rate_array" frequency
    """
    ## time_array in ms !!
    # so multplying rate array

    Npop_pre = AFF_TO_POP_MATRIX.shape[0] # 
    
    true_indices, true_times = [], []
    DT = (time_array[1]-time_array[0])
    
    # trivial way to generate inhomogeneous poisson events
    for it in range(len(time_array)):
        rdm_num = np.random.random(Npop_pre)
        for ii in np.arange(Npop_pre)[rdm_num<DT*rate_array[it]*1e-3]:
            true_indices.append(ii)
            true_times.append(time_array[it])

    indices, times = np.empty(0, dtype=np.int), np.empty(0, dtype=np.float64)
    for ii, tt in zip(true_indices, true_times):
        indices = np.concatenate([indices, np.array(AFF_TO_POP_MATRIX[ii,:], dtype=int)]) # all the indices
        times = np.concatenate([times, np.array([tt for j in range(len(AFF_TO_POP_MATRIX[ii,:]))])])

    # because brian2 can not handle multiple spikes in one bin, we shift them by dt when concomitant
    indices, times, success = deal_with_multiple_spikes_within_one_bin(indices, times, DT)
    if not success:
        indices, times, success = deal_with_multiple_spikes_within_one_bin(indices, times, DT)
                    
    return indices, times*brian2.ms, np.array(true_indices), np.array(true_times)*brian2.ms

def construct_feedforward_input_correlated(NTWK, target_pop,
                                           afferent_pop,\
                                           t, rate_array,\
                                           with_presynaptic_spikes=False,
                                           AFF_TO_POP_MATRIX=None,
                                           verbose=False,
                                           SEED=1):
    """
    POPS and AFFERENCE_ARRAY should be 1D arrrays as their is only one 
    source population

    'pop_for_conductance' is the string identifying the source conductance
    that will be incremented by the afferent input !!
    """

    np.random.seed(SEED) # setting the seed !

    Model = NTWK['Model']
    # extract parameters of the afferent input
    pconn = Model['p_'+afferent_pop+'_'+target_pop]
    Qsyn = Model['Q_'+afferent_pop+'_'+target_pop]

    #finding the target pop in the brian2 objects
    ipop = np.argwhere(NTWK['POPULATIONS']==target_pop).flatten()[0]
    
    if verbose:
        print('drawing Poisson process for afferent spikes [...]')
    if AFF_TO_POP_MATRIX is None:
        AFF_TO_POP_MATRIX = np.array([\
                np.random.choice(np.arange(NTWK['POPS'][ipop].N), int(pconn*NTWK['POPS'][ipop].N), replace=False)\
                for k in range(Model['N_'+afferent_pop])])

    indices, times, true_indices, true_times = set_spikes_from_time_varying_rate_correlated(\
                                                                  t, rate_array, AFF_TO_POP_MATRIX,\
                                                                  SEED=(SEED+2)**2%100)

    pre_increment = 'G'+afferent_pop+target_pop+' += w'
    spikes = brian2.SpikeGeneratorGroup(NTWK['POPS'][ipop].N, indices, times, sorted=False)
    synapse = brian2.Synapses(spikes, NTWK['POPS'][ipop], on_pre=pre_increment,\
                              model='w:siemens')
    
    synapse.connect('i==j')
    synapse.w = Qsyn*brian2.nS
    
    NTWK['PRE_SPIKES'].append(spikes)
    NTWK['PRE_SYNAPSES'].append(synapse)
    
    if with_presynaptic_spikes:
        if 'iRASTER_PRE' in NTWK.keys():
            NTWK['iRASTER_PRE'].append(indices)
            NTWK['tRASTER_PRE'].append(times)
        else: # we create the key
            NTWK['iRASTER_PRE'] = [indices]
            NTWK['tRASTER_PRE'] = [times]

        if 'iRASTER_PRE_in_terms_of_Pre_Pop' in NTWK.keys():
            NTWK['iRASTER_PRE_in_terms_of_Pre_Pop'].append(true_indices)
            NTWK['tRASTER_PRE_in_terms_of_Pre_Pop'].append(true_times)
        else: # we create the key
            NTWK['iRASTER_PRE_in_terms_of_Pre_Pop'] = [true_indices]
            NTWK['tRASTER_PRE_in_terms_of_Pre_Pop'] = [true_times]
            
    return AFF_TO_POP_MATRIX

def set_spikes_from_time_varying_rate_synchronous(time_array, rate_array,
                                                  DUPLICATION_MATRIX, AFF_TO_POP_MATRIX, SEED=1):
    """
    here, we don't assume that all inputs are decorrelated, we actually
    model a population of N neurons and just produce spikes according
    to the "rate_array" frequency
    """
    ## time_array in ms !!
    # so multplying rate array

    np.random.seed(SEED+18) # setting the seed !
    N_independent = DUPLICATION_MATRIX.shape[0] # 
    
    true_indices, true_times = [], []
    DT = (time_array[1]-time_array[0])
    
    # trivial way to generate inhomogeneous poisson events
    for it in range(len(time_array)):
        rdm_num = np.random.random(N_independent)
        for ii in np.arange(N_independent)[rdm_num<DT*rate_array[it]*1e-3/N_independent]: # need to divide by duplicated events
            for jj in DUPLICATION_MATRIX[ii, :]:
                true_indices.append(jj)
                true_times.append(time_array[it])
            
    indices, times = np.empty(0, dtype=np.int), np.empty(0, dtype=np.float64)
    for ii, tt in zip(true_indices, true_times):
        indices = np.concatenate([indices, np.array(AFF_TO_POP_MATRIX[ii,:], dtype=int)]) # all the indices
        times = np.concatenate([times, np.array([tt for j in range(len(AFF_TO_POP_MATRIX[ii,:]))])])

    # because brian2 can not handle multiple spikes in one bin, we shift them by dt when concomitant
    indices, times, success = deal_with_multiple_spikes_within_one_bin(indices, times, DT)
    if not success:
        indices, times, success = deal_with_multiple_spikes_within_one_bin(indices, times, DT)
                    
    return indices, times*brian2.ms, np.array(true_indices), np.array(true_times)

def construct_feedforward_input_synchronous(NTWK, target_pop,
                                            afferent_pop,\
                                            N_source, N_target, N_duplicate,
                                            t, rate_array,\
                                            conductanceID='AA',\
                                            with_presynaptic_spikes=False,
                                            AFF_TO_POP_MATRIX=None,
                                            SEED=1):
    """
    POPS and AFFERENCE_ARRAY should be 1D arrrays as their is only one 
    source population

    'pop_for_conductance' is the string identifying the source conductance
    that will be incremented by the afferent input !!
    """

    np.random.seed(SEED) # setting the seed !

    N_independent = int(N_source/N_duplicate)
    N_source = N_independent*N_duplicate # N_source needs to be a multiple of N_duplicate

    DUPLICATION_MATRIX = np.array([\
                                  np.random.choice(np.arange(N_source), N_duplicate, replace=False)\
                                  for k in range(N_independent)])
    
    AFF_TO_POP_MATRIX = np.array([\
                                  np.random.choice(np.arange(N_target), int(afferent_pop['pconn']*N_target), replace=False)\
                                  for k in range(N_source)])
    
    indices, times, true_indices, true_times = set_spikes_from_time_varying_rate_synchronous(\
                                                        t, rate_array,\
                                                        DUPLICATION_MATRIX, AFF_TO_POP_MATRIX,\
                                                        SEED=(SEED+2)**2%100)

    spikes = brian2.SpikeGeneratorGroup(NTWK['POPS'][ipop].N, indices, times, sorted=False)
    synapse = brian2.Synapses(spikes, NTWK['POPS'][ipop], on_pre='G'+conductanceID+' += w',\
                              model='w:siemens')
    synapse.connect('i==j')
    synapse.w = afferent_pop['Q']*brian2.nS
    
    NTWK['PRE_SPIKES'].append(spikes)
    NTWK['PRE_SYNAPSES'].append(synapse)
    
    if with_presynaptic_spikes:
        if 'iRASTER_PRE' in NTWK.keys():
            NTWK['iRASTER_PRE'].append(indices)
            NTWK['tRASTER_PRE'].append(times)
        else: # we create the key
            NTWK['iRASTER_PRE'] = [indices]
            NTWK['tRASTER_PRE'] = [times]

        if 'iRASTER_PRE_in_terms_of_Pre_Pop' in NTWK.keys():
            NTWK['iRASTER_PRE_in_terms_of_Pre_Pop'].append(true_indices)
            NTWK['tRASTER_PRE_in_terms_of_Pre_Pop'].append(true_times)
        else: # we create the key
            NTWK['iRASTER_PRE_in_terms_of_Pre_Pop'] = [true_indices]
            NTWK['tRASTER_PRE_in_terms_of_Pre_Pop'] = [true_times]
            
    return AFF_TO_POP_MATRIX

            
