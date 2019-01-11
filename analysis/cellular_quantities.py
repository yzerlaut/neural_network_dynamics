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

def get_firing_rate(data, pop='Exc',
                    tdiscard=200):

    FR = []
    for i in range(len(data['VMS_'+pop])):
        tspikes = data['tRASTER_'+str(pop)][np.argwhere(data['iRASTER_'+str(pop)]==i).flatten()]
        FR.append(len(tspikes[tspikes>tdiscard])/(data['tstop']-tdiscard))
    return 1e3*np.array(FR) # from ms to s -> Hz

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
        if len(data['VMS_'+pop][i][cond])>1:
            MUV.append(np.mean(data['VMS_'+pop][i][cond]))
            SV.append(np.std(data['VMS_'+pop][i][cond]))
            SKV.append(skew(data['VMS_'+pop][i][cond]))
            try:
                TV.append(get_acf_time(data['VMS_'+pop][i][cond], data['dt'], min_time=1., max_time=100., procedure='integrate'))
            except IndexError:
                print('problem in determining Tv ...')
        else:
            print('--------------------------------------')
            print('no subthreshold dynamics in those cell...     ', pop)
            print('--------------------------------------')
        
    return np.array(MUV), np.array(SV), np.array(SKV), np.array(TV)


def get_synaptic_currents(data, pop='Exc', tdiscard=200):

    """
    TO BE WRITTEN 

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

    TO BE DONE !
    make this function more general ! (here assumes RecExc and RecInh names)

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
    """


if __name__=='__main__':
    import neural_network_dynamics.main as ntwk
    data = ntwk.load_dict_from_hdf5('../../sparse_vs_balanced/data/weakrec_level2.h5')
    print(get_firing_rate(data, pop='RecExc'))
    print(get_Vm_fluct_props(data, pop='RecExc'))



