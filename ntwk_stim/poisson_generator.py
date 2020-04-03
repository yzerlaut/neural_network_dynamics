"""
Stimulate a neuron population with Poisson process (homogeneous or inhomogeneous)
"""
import numpy as np

def spikes_from_time_varying_rate(time_array, rate_array,
                                  N=100,
                                  Nsyn=10,
                                  SEED=1):
    """
    GENERATES a POISSON INPUT TO POST_SYNAPTIC CELLS
    
    2 modes:
    - One where we mimick an afferent population with a 
      real connectivity pattern to the post-syn pop.  
               -> set a AFF_TO_POP_MATRIX
    - One where there is not explicit connectivity pattern
      and we use the property of poisson processes to 
      set the input freq.!
               -> set a Nsyn
    """
    np.random.seed(SEED) # setting the seed !
    
    ## time_array in ms !!
    # so multplying rate array
    DT = (time_array[1]-time_array[0])

    # indices and spike times for the post-synaptic cell:
    indices, times = [], []
    # indices and spike times in terms of the pre-synaptic cell
    # (N.B. valid only if AFF_TO_POP_MATRIX is not None)
    pre_indices, pre_times = [], []

    # if AFF_TO_POP_MATRIX is not None:

    #     Npop_pre = AFF_TO_POP_MATRIX.shape[0] #
    #     # trivial way to generate inhomogeneous poisson events
    #     for it in range(len(time_array)):
    #         rdm_num = np.random.random(Npop_pre)
    #         for ii in np.arange(Npop_pre)[rdm_num<DT*rate_array[it]*1e-3]:
    #             pre_indices.append(ii)
    #             pre_times.append(time_array[it])
    #     # and then distribute it across the post-synaptic cells
    #     indices, times = np.empty(0, dtype=np.int), np.empty(0, dtype=np.float64)
    #     for ii, tt in zip(pre_indices, pre_times):
    #         indices = np.concatenate([indices, np.array(AFF_TO_POP_MATRIX[ii,:], dtype=int)]) # all the indices
    #         times = np.concatenate([times, np.array([tt for j in range(len(AFF_TO_POP_MATRIX[ii,:]))])])


    # trivial way to generate inhomogeneous poisson events
    for it in range(len(time_array)):
        rdm_num = np.random.random(N)
        for ii in np.arange(N)[rdm_num<DT*Nsyn*rate_array[it]*1e-3]:
            indices.append(ii) # all the indices
            times.append(time_array[it]) # all the same time !
                
    return np.array(indices), np.array(times), np.array(pre_indices), np.array(pre_times)


def deal_with_multiple_spikes_per_bin(indices, times, t, verbose=False, debug=False):
    """
    Brian2 constraint:
    spikes have to be shifted to insure no overlapping presynaptic spikes !
    """
    dt = t[1]-t[0]
    
    if verbose:
        print('Insuring only 1 presynaptic spikes per dt [...]')

    indices2, times2 = np.empty(0, dtype=int), np.empty(0) 
    for nn in np.array(np.unique(indices), dtype=int):
        if debug:
            print('neuron ', nn)
        binned_spikes = np.histogram(times[nn==indices], bins=t)[0]
        new_binned_spikes = 0.*binned_spikes
        range_of_spk_num = np.arange(1, np.max(binned_spikes)+1)
        for spk_num in range_of_spk_num[::-1]:
            if debug:
                print(spk_num, binned_spikes)
            # let's find the empty ones
            iempty = np.argwhere(new_binned_spikes==0).flatten()
            # let's find the times corresponding to this high spike number:
            for jj in np.argwhere(binned_spikes==spk_num).flatten():
                new_binned_spikes[iempty[np.argmin((iempty-jj)**2)]] += 1
                binned_spikes[jj] -= 1
            if debug:
                print(spk_num, binned_spikes, new_binned_spikes)

        times2 = np.concatenate([times2, np.array(t[:-1][new_binned_spikes==1]+dt/2.)])
        indices2 = np.concatenate([indices2, nn*np.ones(len(t[:-1][new_binned_spikes==1]), dtype=int)])

    return indices2, times2

    
if __name__=='__main__':

    tstop, dt = 2e3, 0.1
    t = np.arange(int(tstop/dt))*dt
    print('generate spikes [...]')
    indices, times, _, _ = spikes_from_time_varying_rate(t, 50+0.*t, 100, 100, SEED=1)
    indices = np.concatenate([indices, np.random.choice(np.arange(100),1000)])
    times = np.concatenate([times, 50*np.ones(1000)])
    indices, times = deal_with_multiple_spikes_per_bin(indices, times, t, verbose=True, debug=False)
    
    # for ii in np.unique(indices):
    #     binned_spikes = np.histogram(times[ii==indices], bins=t)[0]
    #     if len(binned_spikes[binned_spikes>1])>0:
    #         print('+1 neuron with duplicate spikes')

    
    print('--------- AFTER ------')
    for ii in np.unique(indices):
        binned_spikes = np.histogram(times[ii==indices], bins=t)[0]
        if len(binned_spikes[binned_spikes>1])>0:
            print('+1 neuron with duplicate spikes')

    # t = np.arange(10)*0.1
    # indices = np.concatenate([np.ones(8)*5, np.ones(4)*3, np.ones(1)])
    # times = np.concatenate([np.ones(8)*.5, np.ones(4)*.3, np.ones(1)*.1])
    # indices, times = deal_with_multiple_spikes_per_bin(indices, times, t, verbose=True, debug=True)

