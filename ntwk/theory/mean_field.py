import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from theory.Vm_statistics import getting_statistical_properties
from theory.probability import Proba_g_P
from theory.spiking_function import firing_rate
from cells.cell_library import built_up_neuron_params
from theory.tf import build_up_afferent_synaptic_input
import numpy as np
from scipy.integrate import odeint

import time

def input_output(neuron_params, SYN_POPS, RATES, COEFFS,
                 current_input = 0, # in pA of not SI
                 already_SI=False):
    muV, sV, gV, Tv = getting_statistical_properties(neuron_params,
                                                     SYN_POPS, RATES,
                                                     current_input = current_input,
                                                     already_SI=already_SI)
    if already_SI:
        Vthre = neuron_params['Vthre']
    else:
        Vthre = 1e-3*neuron_params['Vthre']
        
    Proba = Proba_g_P(muV, sV, gV, Vthre)
    Fout = firing_rate(muV, sV, gV, Tv, Proba, COEFFS)
    return Fout

def solve_mean_field_first_order(Model,
                                 DYN_SYSTEM = {
                                     'RecExc': {'aff_pops':['AffExc'], 'x0':1.},
                                     'RecInh': {'aff_pops':['AffExc'], 'x0':1.}
                                 },
                                 INPUTS = {'AffExc_RecExc':np.ones(1000),
                                           'AffExc_RecInh':np.ones(1000)},
                                 CURRENT_INPUTS = {},
                                 dt = 1e-4, tstop = .1,
                                 T=5e-3,
                                 replace_x0=False,
                                 verbose=True):
    """
    

    if replace_x0 -> then you can directly use get_stat_props
    """
    DYN_KEYS = [key for key in DYN_SYSTEM.keys()] # for a quick access to the dynamical system variables

    # initialize neuronal and synaptic params
    for key in DYN_KEYS:
        DYN_SYSTEM[key]['nrn_params'] = built_up_neuron_params(Model, key)
        DYN_SYSTEM[key]['syn_input'] = build_up_afferent_synaptic_input(Model,\
                                                        DYN_KEYS+DYN_SYSTEM[key]['aff_pops'], key,
                                                        verbose=verbose)


    # --- CONSTRUCT THE DIFFERENTIAL OPERATOR --- #
    def dX_dt(X, t, dt, DYN_KEYS, DYN_SYSTEM, INPUTS, CURRENT_INPUTS):
        dX_dt, RATES = [], {}
        # we fill the X-defined recurent act:
        for x, key in zip(X, DYN_KEYS):
            RATES['F_'+key] = x
        # then we compute it, key by key
        for i, key in enumerate(DYN_KEYS):
            for aff_key in DYN_SYSTEM[key]['aff_pops']:
                RATES['F_'+aff_key] = INPUTS[aff_key+'_'+key][\
                                        min([int(t/dt), len(INPUTS[aff_key+'_'+key])-1]) ]
            if key in CURRENT_INPUTS:
                current_input = CURRENT_INPUTS[key][min([int(t/dt), len(CURRENT_INPUTS[key])-1])]
            else:
                current_input = 0
            Fout = input_output(DYN_SYSTEM[key]['nrn_params'],
                                DYN_SYSTEM[key]['syn_input'],
                                RATES,
                                Model['COEFFS_'+key],
                                current_input = current_input)
            dX_dt.append((Fout-X[i])/T) # Simple one-dimensional framework
        return dX_dt
    # ------------------------------------------- #

    # starting point
    X0 = []
    for key in DYN_KEYS:
        X0.append(DYN_SYSTEM[key]['x0'])

    if verbose:
        print('running ODE integration [...]')
        start_time = time.time()
    X = odeint(dX_dt, X0, np.arange(int(tstop/dt))*dt,
               args=(dt, DYN_KEYS, DYN_SYSTEM, INPUTS, CURRENT_INPUTS))
    if verbose:
        print("--- ODE integration took %.1f seconds ---" % (time.time() - start_time))

    output = {}
    for key, x in zip(DYN_KEYS, X.T):
        output[key] = x
        if replace_x0:
            DYN_SYSTEM[key]['x0'] = x[-1]
    return output

def find_fp(Model,
            DYN_SYSTEM = {
                'RecExc': {'aff_pops':['AffExc'], 'x0':1.},
                'RecInh': {'aff_pops':['AffExc'], 'x0':1.}
            },
            INPUTS = {'AffExc_RecExc':np.ones(1000),
                      'AffExc_RecInh':np.ones(1000)},
            CURRENT_INPUTS = {'RecExc':np.ones(1000)}, # very low current just for demo
            dt=0.1,
            T=5e-3,
            tstop=None,
            replace_x0=True):
    
    if tstop is None:
        tstop = 50.*T

    def func(t, aff_key, target_key):
        return DYN_SYSTEM[target_key]['aff_pops_input_values'][0]

    for key in DYN_SYSTEM.keys():
        for aff_key in DYN_SYSTEM[key]['aff_pops']:
            if type(INPUTS[aff_key+'_'+key]) is float:
                INPUTS[aff_key+'_'+key] = INPUTS[aff_key+'_'+key]*np.ones(int(tstop/dt))
            elif len(INPUTS[aff_key+'_'+key])>1:
                print('Fixed point evaluated at the Input level %s->%s: %.2fHz' % (aff_key, key, INPUTS[aff_key+'_'+key][0]))
                INPUTS[aff_key+'_'+key] = INPUTS[aff_key+'_'+key][0]*np.ones(int(tstop/dt))
            else:
                INPUTS[aff_key+'_'+key] = INPUTS[aff_key+'_'+key][0]*np.ones(int(tstop/dt))

    X = solve_mean_field_first_order(Model, DYN_SYSTEM,
                                     INPUTS=INPUTS,
                                     dt=dt, tstop=tstop, T=T, replace_x0=replace_x0)

    if replace_x0:
        # we set the initial condition 'x0' with 
        for key in DYN_SYSTEM.keys():
            DYN_SYSTEM[key]['x0'] = X[key][-1]
    return [X[key][-1] for key in DYN_SYSTEM.keys()]

def get_full_statistical_quantities(Model, DYN_SYSTEM, RATES,
                                    CURRENT_INPUTS = {}):
                                    
    """
    DYN_SYSTEM = {'RecExc': {'aff_pops':['AffExc'], 'x0':1.},
                  'RecInh': {'aff_pops':['AffExc'], 'x0':1.}}    
    RATES = {'F_AffExc':1.,
             'F_RecExc':1.,
             'F_RecInh':1.}
    """

    DYN_KEYS = [key for key in DYN_SYSTEM.keys()] # for a quick access to the dynamical system variables

    # initialize neuronal and synaptic params (very likely already done)
    for key in DYN_KEYS:
        DYN_SYSTEM[key]['nrn_params'] = built_up_neuron_params(Model, key)
        DYN_SYSTEM[key]['syn_input'] = build_up_afferent_synaptic_input(Model,
                                                DYN_KEYS+DYN_SYSTEM[key]['aff_pops'], key)
        
    # output, RATES = {}, {}
    # for key in DYN_KEYS:
    #     RATES['F_'+key] = DYN_SYSTEM[key]['x0'] # we use x0 as the activity point to compute
    output = {}
    for key in DYN_KEYS:
        current_input = 0.
        if key in CURRENT_INPUTS:
            current_input = CURRENT_INPUTS[key]
            
        output['muV_'+key], output['sV_'+key],\
            output['gV_'+key], output['Tv_'+key],\
            output['Isyn_'+key] = getting_statistical_properties(
                DYN_SYSTEM[key]['nrn_params'],
                DYN_SYSTEM[key]['syn_input'],
                RATES,
                current_input=current_input,
                already_SI=False, with_Isyn=True)
    return output

if __name__=='__main__':

    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from graphs.my_graph import graphs
    mg = graphs()


    sys.path.append('../configs/The_Spectrum_of_Asynch_Dyn_2018/')
    from model import Model

    Model['COEFFS_RecExc'] = np.load('../configs/The_Spectrum_of_Asynch_Dyn_2018/COEFFS_RecExc.npy')
    Model['COEFFS_RecInh'] = np.load('../configs/The_Spectrum_of_Asynch_Dyn_2018/COEFFS_RecInh.npy')

    tstop, dt = 1, 5e-4

    DYN_SYSTEM = {
        'RecExc': {'aff_pops':['AffExc'], 'x0':1.},
        'RecInh': {'aff_pops':['AffExc'], 'x0':1.}
    }
    INPUTS = {
        'AffExc_RecExc':np.ones(int(tstop/dt))*3.,
        'AffExc_RecInh':np.ones(int(tstop/dt))*3.
        }
    CURRENT_INPUTS = {'RecExc':np.ones(1000)}, # very low current just for demo
        
    
    X0 = find_fp(Model, DYN_SYSTEM,
                 INPUTS=INPUTS,
                 CURRENT_INPUTS=CURRENT_INPUTS,
                 dt=dt,
                 tstop=tstop,
                 replace_x0=True)
    X = solve_mean_field_first_order(Model, DYN_SYSTEM,
                                     INPUTS=INPUTS,
                                     CURRENT_INPUTS=CURRENT_INPUTS,
                                     dt=dt, tstop=tstop)

    print(get_full_statistical_quantities(Model,
                                          DYN_SYSTEM,
                                          {'F_AffExc':3., 'F_RecExc':X['RecExc'][-1], 'F_RecInh':X['RecInh'][-1]},
                                          CURRENT_INPUTS=CURRENT_INPUTS))
    
    fig, ax = mg.figure()
    ax.set_yscale('log')
    mg.plot(Y=[X['RecExc'], X['RecInh']], COLORS=[mg.g, mg.r], ax=ax, axes_args=dict(yticks=[0.1,1.,10], ylim=[1e-5, 1000]))
    mg.show()
