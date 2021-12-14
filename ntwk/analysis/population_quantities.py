import numpy as np
from itertools import combinations # for cross correlations
from scipy.stats import skew
import elephant, neo, quantities # eletrophysiology analysis toolkit

# for smoothing
from scipy.ndimage.filters import gaussian_filter1d

from analyz.processing.signanalysis import get_acf_time


def gaussian_smoothing(signal, idt_sbsmpl=10):
    return gaussian_filter1d(signal, idt_sbsmpl)


def smooth_population_activity(rate_array, dt, Tsmoothing):
    if Tsmoothing>2*dt:
        return gaussian_smoothing(rate_array, int(Tsmoothing/dt))
    else:
        print('/!\ SMOOTHING AT THAT TIME SCALE UNPOSSIBLE, dt is %s and Tsmoothing is %s' % (dt, Tsmoothing))
        return rate_array

    
def get_CV_spiking(data, pop='Exc'):
    """see Kumar et al. 2008"""
    ispikes, tspikes = data['iRASTER_'+pop], data['tRASTER_'+pop]
    CV = []
    for i in np.unique(ispikes):
        tspikes_i = tspikes[np.argwhere(ispikes==i).flatten()]
        isi = np.diff(tspikes_i)
        if len(isi)>2:
            CV.append(np.std(isi)/np.mean(isi))
    if len(CV)>1:
        return np.array(CV).mean()
    else:
        return 0

def get_synchrony_of_spiking(data, pop='Exc',
                             Tbin=2,
                             method='spk-trains-cross-correl',
                             Nmax_pairs=4000,
                             tzoom=[-np.inf, np.inf],
                             seed=23):
    """
    Different methods:

    - Spike Train Cross-Correlation: 'spk-trains-cross-correl'
            * see Kumar et al. 2008
    - Spike Time Tiling Coefficient
            * Cutts & Eglen. Detecting pairwise correlations in spike trains: an objective comparison of methods and application to the study of retinal waves. 
              Journal of Neuroscience, 34(43):14288–14303, 2014

    we introduce a limiting number of pairs for fast computation"""

    np.random.seed(seed)

    n, Ntot = 0, 0
    while str(n) in data.keys():
        if (data[str(n)]['name']==pop):
            Ntot = int(data[str(n)]['N'])
        n+=1
    if Ntot==0:
        print('key not recognized in neural_net_dyn.macro_quantities.get_synchrony_of_spiking !!')

    ispikes, tspikes = data['iRASTER_'+pop], data['tRASTER_'+pop]
    # in case we focus on a subset of the temporal dynamics (none by default)
    cond = (tspikes>tzoom[0]) & (tspikes<tzoom[1])
    ispikes, tspikes = ispikes[cond], tspikes[cond]
    
    ispikes_unique = np.unique(ispikes)
    new_t = np.arange(int(data['tstop']/Tbin)+5)*Tbin

    if len(ispikes_unique)>3:
        # if there are at least two couples
        couples = list(combinations(ispikes_unique, r=2))
        Nmax_pairs = min([len(couples), Nmax_pairs])
        rdm_picks = np.random.choice(range(len(couples)), Nmax_pairs)

        SYNCH = []
        for r in range(Nmax_pairs):
            i, j = couples[rdm_picks[r]]
            tspikes_i = tspikes[np.argwhere(ispikes==i).flatten()]
            tspikes_j = tspikes[np.argwhere(ispikes==j).flatten()]
            if method=='spk-trains-cross-correl':
                spk_train_i, _ = np.histogram(tspikes_i, bins=new_t)
                spk_train_j, _ = np.histogram(tspikes_j, bins=new_t)
                SYNCH.append(np.corrcoef(spk_train_i, spk_train_j)[0,1])
            elif method in ['spk-time-tiling-coef', 'STTC']:
                spk_train_i = neo.SpikeTrain(tspikes_i, t_stop=data['tstop'], units='ms')
                spk_train_j = neo.SpikeTrain(tspikes_j, t_stop=data['tstop'], units='ms')
                SYNCH.append(elephant.spike_train_correlation.spike_time_tiling_coefficient(spk_train_i, spk_train_j, dt=Tbin*quantities.ms))
            else:
                SYNCH.append(0)
                
        return np.array(SYNCH).mean()
    else:
        return 0

    
def get_mean_pop_act(data, pop='Exc', tdiscard=200):
    
    t = np.arange(int(data['tstop']/data['dt']))*data['dt']
    cond = t>tdiscard
    return data['POP_ACT_'+pop][cond].mean()

def get_std_pop_act(data, pop='Exc', tdiscard=200):
    
    t = np.arange(int(data['tstop']/data['dt']))*data['dt']
    cond = t>tdiscard
    return data['POP_ACT_'+pop][cond].std()


def get_currents_and_balance(data, pop='Exc', tdiscard=200, Vreset=-70):
    
    t = np.arange(int(data['tstop']/data['dt']))*data['dt']
    
    meanIe, meanIi = 0, 0
    for i in range(len(data['ISYNe_'+pop])):
        # discarding initial transient as well as refractory period !
        cond = (t>tdiscard) & (data['VMS_'+pop][i,:]!=Vreset)
        meanIe += data['ISYNe_'+pop][i,cond].mean()/len(data['ISYNe_'+pop])
        meanIi += data['ISYNi_'+pop][i,cond].mean()/len(data['ISYNi_'+pop])

    # meanIe, meanIi = data['ISYNe_'+pop][:,cond].mean(), data['ISYNi_'+pop][:,cond].mean()
    # print(meanIe, meanIi)
    if meanIe>0:
        balance = -meanIi/meanIe
    else:
        balance = 0
    return meanIe, meanIi, balance

def get_afferent_and_recurrent_currents(data, pop='Exc', tdiscard=200, Vreset=-70):
    """
    TO BE DONE !
    make this function more general ! (here assumes RecExc and RecInh names)
    """
    t = np.arange(int(data['tstop']/data['dt']))*data['dt']
    
    meanIe_Aff, meanIe_Rec, meanIi_Rec = 0, 0, 0
    for i in range(len(data['VMS_'+pop])):
        # discarding initial transient as well as refractory period !
        cond = (t>tdiscard) & (data['VMS_'+pop][i,:]!=Vreset)

        # meanIi += data['ISYNi_'+pop][i,cond].mean()/len(data['ISYNi_'+pop])
        meanIe_Rec += np.abs(np.mean(data['POP_ACT_RecExc'])*data['p_RecExc_'+pop]*data['N_RecExc']*\
                        data['Q_RecExc_'+pop]*data['Tse']*1e-3*(data['Ee']-np.mean(data['VMS_'+pop][i,cond])))/len(data['VMS_'+pop])
        meanIi_Rec += np.abs(np.mean(data['POP_ACT_RecInh'])*data['p_RecInh_'+pop]*data['N_RecInh']*\
                              data['Q_RecInh_'+pop]*data['Tsi']*1e-3*(data['Ei']-np.mean(data['VMS_'+pop][i,cond])))/len(data['VMS_'+pop])
        meanIe_Aff += np.abs(data['F_AffExc']*data['p_AffExc_'+pop]*data['N_AffExc']*\
                     data['Q_AffExc_'+pop]*data['Tse']*1e-3*(data['Ee']-np.mean(data['VMS_'+pop][i,cond])))/len(data['VMS_'+pop])

    return meanIe_Aff, meanIe_Rec, meanIi_Rec


def get_all_macro_quant(data, exc_pop_key='Exc', inh_pop_key='Inh', other_pops=[]):
    """
    need to make this function more general !
    """

    output = {}
    # weighted sum (by num of neurons) over exc and inhibtion
    output['synchrony'] = .2*get_synchrony_of_spiking(data, pop=inh_pop_key)+\
                          .8*get_synchrony_of_spiking(data, pop=exc_pop_key)
    output['irregularity'] = .2*get_CV_spiking(data, pop=inh_pop_key)+.8*get_CV_spiking(data, pop=exc_pop_key)
    output['meanIe_'+exc_pop_key], output['meanIi_'+exc_pop_key], output['balance_'+exc_pop_key] = get_currents_and_balance(data,
                                                                                                        pop=exc_pop_key)
    output['meanIe_'+inh_pop_key], output['meanIi_'+inh_pop_key], output['balance_'+inh_pop_key] = get_currents_and_balance(data,
                                                                                                        pop=inh_pop_key)
    
    output['meanIe_Aff'], output['meanIe_Rec'], output['meanIi_Rec'] = get_afferent_and_recurrent_currents(data,
                                                                                                           pop=exc_pop_key)

    try:
        output['mean_exc'] = get_mean_pop_act(data, pop=exc_pop_key)
        output['mean_inh'] = get_mean_pop_act(data, pop=inh_pop_key)
    except KeyError:
        output['mean_exc'] = 0.
        output['mean_inh'] = 0.
        
    for pop in other_pops:
        try:
            output['mean_'+pop] = get_mean_pop_act(data, pop=pop)
        except KeyError:
            output['mean_'+pop] = 0.
            
    return output

if __name__=='__main__':
    import sys
    sys.path.append('../../')
    # from params_scan.aff_exc_aff_dsnh_params_space import get_scan
    import neural_network_dynamics.main as ntwk
    args, F_aff, F_dsnh, DATA = ntwk.get_scan({}, filename='../../params_scan/data/scan.zip')
    print(get_synchrony_of_spiking(DATA[2]))
    print(get_synchrony_of_spiking(DATA[-1]))
    # print(get_CV_spiking(data))
    # print(get_mean_pop_act(data))
    # print(get_mean_pop_act(data, pop='Inh'))
    for data in DATA[8:]:
        print(get_currents_and_balance(data, pop='Exc'))
    # print(get_currents_and_balance(DATA[-1], pop='Inh'))


