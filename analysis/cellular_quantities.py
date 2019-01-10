import numpy as np
import matplotlib.pylab as plt
from itertools import combinations # for cross correlations
import sys, pathlib
from scipy.stats import skew
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
try:
    from data_analysis.processing.signanalysis import get_acf_time
except ImportError:
    print('---------------------------------------------------------------')
    print('you need the data_analysis folder')
    print('get it at: bitbucket.org/yzerlaut/data_analysis')
    print('---------------------------------------------------------------')

def get_synchrony_of_spiking(data, pop='Exc',
                             Tbin=2, Nmax_pairs=4000,
                             seed=23, tzoom=[-np.inf, np.inf]):
    """see Kumar et al. 2008

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
            spk_train_i, _ = np.histogram(tspikes_i, bins=new_t)
            spk_train_j, _ = np.histogram(tspikes_j, bins=new_t)
            SYNCH.append(np.corrcoef(spk_train_i, spk_train_j)[0,1])
        return np.array(SYNCH).mean()
    else:
        return 0
    
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


def get_firing_rate(data, pop='Exc',
                    tdiscard=200):

    FR = []
    for i in range(len(data['VMS_'+pop])):
        tspikes = data['tRASTER_'+str(pop)][np.argwhere(data['iRASTER_'+str(pop)]==i).flatten()]
        FR.append(len(tspikes[tspikes>tdiscard])/(data['tstop']-tdiscard))
    return np.array(FR)

def get_Vm_fluct_props(data, pop='Exc',
                       tdiscard=200,
                       twindow=None):

    if twindow is None:
        try:
            twindow = data[pop+'_Trefrac']
        except KeyError:
            print('Refractory period not found, set as 5ms')
            twindow = 5
            
    t = np.arange(int(data['tstop']/data['dt']))*data['dt']

    MUV, SV, SKV, TV = [], [], [], []
    
    for i in range(len(data['VMS_'+pop])):
        cond = (t>tdiscard)
        # then removing spikes
        tspikes = data['tRASTER_'+str(pop)][np.argwhere(data['iRASTER_'+str(pop)]==i).flatten()]
        for ts in tspikes:
            cond = cond & np.invert((t>=ts-twindow) & (t<=(ts+twindow)))
        MUV.append(data['VMS_'+pop][i][cond].mean())
        SV.append(data['VMS_'+pop][i][cond].std())
        SKV.append(skew(data['VMS_'+pop][i][cond]))
        TV.append(get_acf_time(data['VMS_'+pop][i][cond], data['dt'], min_time=1., max_time=100., procedure='integrate'))
        
    return np.array(MUV), np.array(SV), np.array(SKV), np.array(TV)


if __name__=='__main__':
    import neural_network_dynamics.main as ntwk
    data = ntwk.load_dict_from_hdf5('../../sparse_vs_balanced/data/weakrec_level2.h5')
    # from graphs.my_graph import *
    print(get_firing_rate(data, pop='RecExc'))
    print(get_Vm_fluct_props(data, pop='RecExc'))
    # plot(data['VMS_RecExc'][0])
    # show()
    # import sys
    # sys.path.append('../../')
    # # from params_scan.aff_exc_aff_dsnh_params_space import get_scan
    # args, F_aff, F_dsnh, DATA = ntwk.get_scan({}, filename='../../params_scan/data/scan.zip')
    # print(get_synchrony_of_spiking(DATA[2]))
    # print(get_synchrony_of_spiking(DATA[-1]))
    # # print(get_CV_spiking(data))
    # # print(get_mean_pop_act(data))
    # # print(get_mean_pop_act(data, pop='Inh'))
    # for data in DATA[8:]:
    #     print(get_currents_and_balance(data, pop='Exc'))
    # # print(get_currents_and_balance(DATA[-1], pop='Inh'))


