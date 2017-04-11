import numpy as np
import matplotlib.pylab as plt
from itertools import combinations # for cross correlations
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from graphs.my_graph import *

def get_CV_spiking(data, pop='Exc'):
    """see Kumar et al. 2008"""
    ispikes, tspikes = data['iRASTER_'+pop], data['tRASTER_'+pop]
    CV = []
    for i in np.unique(ispikes):
        tspikes_i = tspikes[np.argwhere(ispikes==i).flatten()]
        isi = np.diff(tspikes_i)
        if len(isi)>2:
            CV.append(np.mean(isi)/np.std(isi))
    if len(CV)>1:
        return np.array(CV).mean()
    else:
        return 0

def get_synchrony_of_spiking(data, pop='Exc', Tbin=2, Nmax_pairs=200):
    """see Kumar et al. 2008

    we introduce a limiting number of pairs for fast computation"""

    n, Ntot = 0, 0
    while str(n) in data.keys():
        if data[str(n)]['name']==pop:
            Ntot = int(data[str(n)]['N'])
        n+=1
    if Ntot==0:
        print('key not recognized !!')
        
    ispikes, tspikes = data['iRASTER_'+pop], data['tRASTER_'+pop]
    SYNCH = []
    ispikes_unique = np.unique(ispikes)
    new_t = np.arange(int(data['tstop']/Tbin))*Tbin

    couples = list(combinations(np.arange(Ntot), r=2))
    rdm_picks = np.random.choice(range(len(couples)), Nmax_pairs)

    for r in range(Nmax_pairs):
        i, j = couples[rdm_picks[r]]
        tspikes_i = tspikes[np.argwhere(ispikes==i).flatten()]
        tspikes_j = tspikes[np.argwhere(ispikes==j).flatten()]
        if len(tspikes_i)>1 and len(tspikes_j)>1:
            spk_train_i, _ = np.histogram(tspikes_i, bins=new_t)
            spk_train_j, _ = np.histogram(tspikes_j, bins=new_t)
            SYNCH.append(np.corrcoef(spk_train_i, spk_train_j)[0,1])
        elif len(tspikes_i)==0 and len(tspikes_j)==0:
            # no spikes is considered as synchronous behavior
            SYNCH.append(1)
            
    if len(SYNCH)>1:
        return np.array(SYNCH).mean()
    else:
        return 1
    
def get_mean_pop_act(data, pop='Exc', tdiscard=200):
    
    t = np.arange(int(data['tstop']/data['dt']))*data['dt']
    cond = t>tdiscard
    return data['POP_ACT_'+pop][cond].mean()

def get_currents_and_balance(data, pop='Exc', tdiscard=200):
    
    t = np.arange(int(data['tstop']/data['dt']))*data['dt']
    cond = t>tdiscard
    meanIe, meanIi = data['ISYNe_'+pop][:,cond].mean(), data['ISYNi_'+pop][:,cond].mean()
    if meanIi<0:
        balance = -meanIe/meanIi
    else:
        balance = 0
    return meanIe, meanIi, balance


def get_all_macro_quant(data):

    output = {}
    output['synchrony'] = get_synchrony_of_spiking(data)
    output['irregularity'] = get_CV_spiking(data)
    output['mean_exc'] = get_mean_pop_act(data, pop='Exc')
    output['mean_inh'] = get_mean_pop_act(data, pop='Inh')
    output['meanIe_Exc'], output['meanIi_Exc'], output['balance_Exc'] = get_currents_and_balance(data, pop='Exc')
    output['meanIe_Inh'], output['meanIi_Inh'], output['balance_Inh'] = get_currents_and_balance(data, pop='Inh')

    return output

    

if __name__=='__main__':
    import sys
    sys.path.append('../../')
    from params_scan.aff_exc_aff_dsnh_params_space import get_scan
    args, F_aff, F_dsnh, DATA = get_scan(\
                    '../../params_scan/data/scan.zip')
    data = DATA[-1]
    print(get_synchrony_of_spiking(data))
    print(get_CV_spiking(data))
    print(get_mean_pop_act(data))
    print(get_mean_pop_act(data, pop='Inh'))
    print(get_currents_and_balance(data, pop='Exc'))
    print(get_currents_and_balance(data, pop='Inh'))


