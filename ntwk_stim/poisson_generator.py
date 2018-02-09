"""

"""
import numpy as np

def spikes_from_time_varying_rate(time_array, rate_array,
                                  N=100,
                                  Nsyn=10,
                                  AFF_TO_POP_MATRIX=None,
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

    if AFF_TO_POP_MATRIX is not None:

        Npop_pre = AFF_TO_POP_MATRIX.shape[0] #
        # trivial way to generate inhomogeneous poisson events
        for it in range(len(time_array)):
            rdm_num = np.random.random(Npop_pre)
            for ii in np.arange(Npop_pre)[rdm_num<DT*rate_array[it]*1e-3]:
                pre_indices.append(ii)
                pre_times.append(time_array[it])
        # and then distribute it across the post-synaptic cells
        indices, times = np.empty(0, dtype=np.int), np.empty(0, dtype=np.float64)
        for ii, tt in zip(pre_indices, pre_times):
            indices = np.concatenate([indices, np.array(AFF_TO_POP_MATRIX[ii,:], dtype=int)]) # all the indices
            times = np.concatenate([times, np.array([tt for j in range(len(AFF_TO_POP_MATRIX[ii,:]))])])

    else:

        # trivial way to generate inhomogeneous poisson events
        for it in range(len(time_array)):
            rdm_num = np.random.random(N)
            for ii in np.arange(N)[rdm_num<DT*Nsyn*rate_array[it]*1e-3]:
                indices.append(ii) # all the indices
                times.append(time_array[it]) # all the same time !
                
    return np.array(indices), np.array(times), np.array(pre_indices), np.array(pre_times)


def deal_with_multiple_spikes_per_bin(indices, times, t, verbose=False):
    """
    Brian2 constraint:
    spikes have to be shifted to insure no overlapping presynaptic spikes !
    """

    if verbose:
        print('Insuring only 1 presynaptic spikes per dt [...]')
        
    # a basic check on the range of firing, to see whether small shifts will work
    binned_spikes = np.histogram(times[0==indices], bins=t)[0]
    if binned_spikes.sum()>.9*len(t):
        print('-------------------------------------------------')
        print('You need to decrease the time step or to reduce the afferent freq. !')
        print('')
        print('-------------------------------------------------')
        return [], []
    else:
        for ii in np.unique(indices):
            nsecurity=0
            binned_spikes = np.histogram(times[ii==indices], bins=t)[0]
            while (np.max(binned_spikes)>1) & (nsecurity<100):
                iempty = np.argwhere(binned_spikes==0).flatten()
                itoomuch = np.argwhere(binned_spikes>1).flatten()
                used_iempty = []
                for jj in itoomuch:
                    iempty_sorted_by_distance = iempty[np.argsort((iempty-jj)**2)]
                    kk=0
                    while (iempty_sorted_by_distance[kk] in used_iempty):
                        kk+=1
                    # now we update the spike time that is problematic:
                    ipb = np.argwhere((indices==ii) & (times>=t[jj])& (times<t[jj+1]))
                    times[ipb[0]] = t[iempty_sorted_by_distance[kk]] # we shift it to an empty bin

                binned_spikes = np.histogram(times[ii==indices], bins=t)[0]
                nsecurity +=1
            if nsecurity>90:
                print('Pb in the shifting of spikes !!')
                
        return indices, times

    
if __name__=='__main__':

    t = np.arange(1000)*0.1
    indices, times, _, _ = spikes_from_time_varying_rate(t, 100+0.*t, 100, 100, SEED=1)
    indices = np.concatenate([indices, np.random.choice(np.arange(100),1000)])
    times = np.concatenate([times, np.random.randn(1000)*10+50])
    
    # t = np.arange(10)*0.1
    # indices, times = np.ones(6), np.ones(6)*.5
    
    for ii in np.unique(indices):
        binned_spikes = np.histogram(times[ii==indices], bins=t)[0]
        if len(binned_spikes[binned_spikes>1])>0:
            print('+1 neuron with duplicate spikes')
            
    indices, times = deal_with_multiple_spikes_per_bin(indices, times)
    print('--------- AFTER ------')
    for ii in np.unique(indices):
        binned_spikes = np.histogram(times[ii==indices], bins=t)[0]
        if len(binned_spikes[binned_spikes>1])>0:
            print('+1 neuron with duplicate spikes')

    # print(times)

