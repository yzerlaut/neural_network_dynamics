import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from theory.Vm_statistics import getting_statistical_properties
from theory.probability import Proba_g_P
from theory.spiking_function import firing_rate
from cells.cell_library import built_up_neuron_params
import numpy as np
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pylab as plt


def build_up_afferent_synaptic_input(Model, POP_STIM, NRN_KEY=None, verbose=True):

    if NRN_KEY is None:
        NRN_KEY = Model['NRN_KEY']

    SYN_POPS = []
    for source_pop in POP_STIM:
        if len(source_pop.split('Exc'))>1:
            Erev, Ts = Model['Ee'], Model['Tse']
        elif len(source_pop.split('Inh'))>1:
            Erev, Ts = Model['Ei'], Model['Tsi']
        else:
            print(' /!\ AFFERENT POP COULD NOT BE CLASSIFIED AS Exc or Inh /!\ ')
            print('-----> set to Exc by default')
            Erev, Ts = Model['Ee'], Model['Tse']

        # here we set 0 connectivity for those not explicitely defined
        try:
            Q, pconn = Model['Q_'+source_pop+'_'+NRN_KEY], Model['p_'+source_pop+'_'+NRN_KEY]
        except KeyError:
            if verbose:
                print('/!\ connection parameters missing, set to 0 by default')
                print(' ---> ', source_pop+'_'+NRN_KEY, 'misses either its Q or pconn argument')
            Q, pconn = 0, 0
            
        SYN_POPS.append({'name':source_pop,
                         'N': Model['N_'+source_pop],
                         'Erev': Erev,
                         'Q': Q, 'pconn': pconn,
                         'Tsyn': Ts})
        # for backward compatibility, only added here
        if 'V0' in Model:
            SYN_POPS[-1]['V0'] = Model['V0']
        if 'alpha_'+source_pop+'_'+NRN_KEY in Model:
            SYN_POPS[-1]['alpha'] = Model['alpha_'+source_pop+'_'+NRN_KEY]
            
    return SYN_POPS
    
def TF(RATES, Model, NRN_KEY):

    neuron_params = built_up_neuron_params(Model, NRN_KEY)
    SYN_POPS = build_up_afferent_synaptic_input(Model, Model['POP_STIM'], NRN_KEY)
    ### OUTPUT OF ANALYTICAL CALCULUS IN SI UNITS !! -> from here SI, be careful...
    muV, sV, gV, Tv = getting_statistical_properties(neuron_params,
                                                     SYN_POPS, RATES,
                                                     already_SI=False)
    Proba = Proba_g_P(muV, sV, gV, 1e-3*neuron_params['Vthre'])
    
    Fout_th = firing_rate(muV, sV, gV, Tv, Proba, Model['COEFFS'])

    return Fout_th



