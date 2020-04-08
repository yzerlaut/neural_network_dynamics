import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pylab as plt
import main as ntwk

import datavyz

################################################################
## ------ Construct populations with their equations -------- ##
## ------------- with recurrent connections ----------------- ##
################################################################

Model = {
    ## ---------------------------------------------------------------------------------
    ### Initialisation by default parameters
    ## UNIT SYSTEM is : ms, mV, pF, nS, pA, Hz (arbitrary and unconsistent, so see code)
    ## ---------------------------------------------------------------------------------
    # numbers of neurons in population
    'N_L23Exc':4000, 'N_PVInh':500, 'N_SOMInh':500, 'N_VIPInh':100, 'N_L4Exc':500,
    # synaptic weights
    'Q_L23Exc_L23Exc':1., 'Q_L23Exc_PVInh':1., 
    'Q_L4Exc_L23Exc':3., 'Q_L4Exc_PVInh':3., 'Q_L4Exc_SOMInh':3., 'Q_L4Exc_VIPInh':3., 
    'Q_PVInh_L23Exc':10., 'Q_PVInh_PVInh':10.,
    'Q_SOMInh_SOMInh':10., 'Q_SOMInh_L23Exc':10.,
    'Q_VIPInh_SOMInh':10.,
    # synaptic time constants
    'Tse':5., 'Tsi':5.,
    # synaptic reversal potentials
    'Ee':0., 'Ei': -80.,
    # connectivity parameters
    'p_L23Exc_L23Exc':0.02, 'p_L23Exc_PVInh':0.02, 'p_L23Exc_SOMInh':0.02, 
    'p_PVInh_L23Exc':0.02, 'p_PVInh_PVInh':0.02, 
    'p_SOMInh_L23Exc':0.02, 'p_SOMInh_SOMInh':0.02, 
    'p_L4Exc_L23Exc':0.1, 'p_L4Exc_PVInh':0.1, 'p_L4Exc_SOMInh':0.1, 'p_L4Exc_VIPInh':0.1,
    'p_VIPInh_SOMInh':0.1,
    # simulation parameters
    'dt':0.1, 'tstop': 2000., 'SEED':3, # low by default, see later
    ## ---------------------------------------------------------------------------------
    # === cellular properties (based on AdExp), population by population ===
    # --> L23Excitatory population (L23Exc, recurrent excitation)
    'L23Exc_Gl':10., 'L23Exc_Cm':200.,'L23Exc_Trefrac':3.,
    'L23Exc_El':-60., 'L23Exc_Vthre':-50., 'L23Exc_Vreset':-60., 'L23Exc_deltaV':0.,
    'L23Exc_a':0., 'L23Exc_b': 0., 'L23Exc_tauw':1e9,
    # --> PVInhibitory population (PVInh, recurrent inhibition)
    'PVInh_Gl':10., 'PVInh_Cm':200.,'PVInh_Trefrac':3.,
    'PVInh_El':-60., 'PVInh_Vthre':-53., 'PVInh_Vreset':-60., 'PVInh_deltaV':0.,
    'PVInh_a':0., 'PVInh_b': 0., 'PVInh_tauw':1e9,
    # --> SOMInhhibitory population (PVInh, recurrent inhibition)
    'SOMInh_Gl':10., 'SOMInh_Cm':200.,'SOMInh_Trefrac':3.,
    'SOMInh_El':-60., 'SOMInh_Vthre':-53., 'SOMInh_Vreset':-60., 'SOMInh_deltaV':0.,
    'SOMInh_a':0., 'SOMInh_b': 0., 'SOMInh_tauw':1e9,
    # --> Disinhibitory population (PVInh, recurrent inhibition)
    'VIPInh_Gl':10., 'VIPInh_Cm':200.,'VIPInh_Trefrac':3.,
    'VIPInh_El':-60., 'VIPInh_Vthre':-53., 'VIPInh_Vreset':-60., 'VIPInh_deltaV':0.,
    'VIPInh_a':0., 'VIPInh_b': 0., 'VIPInh_tauw':1e9,
}


if sys.argv[-1]=='plot':
    # ######################
    # ## ----- Plot ----- ##
    # ######################
    
    ## load file
    data = ntwk.load_dict_from_hdf5('sinusoidal_input_data.h5')
    data['iRASTER_L4Exc'] = data['iRASTER_PRE1']
    data['tRASTER_L4Exc'] = data['tRASTER_PRE1']
    # # ## plot
    fig, _ = ntwk.activity_plots(data,
                                 POP_KEYS = ['L23Exc', 'PVInh', 'SOMInh', 'VIPInh'],
                                 COLORS = ['green', 'red', 'orange', 'purple'],
                                 smooth_population_activity=10.)
    plt.show()
else:
    NTWK = ntwk.build_populations(Model, ['L23Exc', 'PVInh', 'SOMInh', 'VIPInh'],
                                  AFFERENT_POPULATIONS=['L4Exc'],
                                  with_raster=True,
                                  with_Vm=4,
                                  with_pop_act=True,
                                  verbose=True)

    ntwk.build_up_recurrent_connections(NTWK, SEED=5, verbose=True)

    #######################################
    ########### AFFERENT INPUTS ###########
    #######################################

    t_array = ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']
    faff = 0.8+0.7*(1-np.cos(2*np.pi*4e-3*t_array))
    faff[t_array<750] = 1.

    # # # afferent excitation onto cortical excitation and inhibition
    for i, tpop in enumerate(['L23Exc', 'PVInh', 'SOMInh', 'VIPInh']): # both on excitation and inhibition
        ntwk.construct_feedforward_input(NTWK, tpop, 'L4Exc',
                                         t_array, faff,
                                         verbose=True,
                                         SEED=int(i)%37)

    ################################################################
    ## --------------- Initial Condition ------------------------ ##
    ################################################################
    ntwk.initialize_to_rest(NTWK)

    #####################
    ## ----- Run ----- ##
    #####################
    network_sim = ntwk.collect_and_run(NTWK, verbose=True)

    ntwk.write_as_hdf5(NTWK, filename='sinusoidal_input_data.h5')
    print('Results of the simulation are stored as:', 'sinusoidal_input_data.h5')
    print('--> Run \"python sinusoidal_input.py plot\" to plot the results')
