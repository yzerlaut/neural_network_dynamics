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
    # 'N_l23Exc':4000, 'N_PVInh':500, 'N_SOMInh':500, 'N_l4Exc':500,
    'N_l23Exc':400, 'N_PVInh':50, 'N_SOMInh':50, 'N_l4Exc':50,
    # synaptic weights
    'Q_l23Exc_l23Exc':1., 'Q_l23Exc_PVInh':1., 
    'Q_l4Exc_l23Exc':3., 'Q_l4Exc_PVInh':3., 'Q_l4Exc_SOMInh':3., 
    'Q_PVInh_l23Exc':10., 'Q_PVInh_PVInh':10.,
    'Q_SOMInh_PVInh':10., 
    # synaptic time constants
    'Tse':5., 'Tsi':5.,
    # synaptic reversal potentials
    'Ee':0., 'Ei': -80.,
    # connectivity parameters
    'p_l23Exc_l23Exc':0.02, 'p_l23Exc_PVInh':0.02, 
    'p_PVInh_l23Exc':0.02, 'p_PVInh_PVInh':0.02, 
    'p_SOMInh_PVInh':0.02, 'p_SOMInh_l23Exc':0.02, 
    'p_l4Exc_l23Exc':0.1, 'p_l4Exc_PVInh':0.1, 'p_l4Exc_SOMInh':0.1, 
    # simulation parameters
    'dt':0.2, 'tstop': 2000., 'SEED':3, # low by default, see later
    ## ---------------------------------------------------------------------------------
    # === cellular properties (based on AdExp), population by population ===
    # --> l23Excitatory population (l23Exc, recurrent excitation)
    'l23Exc_Gl':10., 'l23Exc_Cm':200.,'l23Exc_Trefrac':3.,
    'l23Exc_El':-60., 'l23Exc_Vthre':-50., 'l23Exc_Vreset':-60., 'l23Exc_deltaV':0.,
    'l23Exc_a':0., 'l23Exc_b': 0., 'l23Exc_tauw':1e9,
    # --> PVInhibitory population (PVInh, recurrent inhibition)
    'PVInh_Gl':10., 'PVInh_Cm':200.,'PVInh_Trefrac':3.,
    'PVInh_El':-60., 'PVInh_Vthre':-53., 'PVInh_Vreset':-60., 'PVInh_deltaV':0.,
    'PVInh_a':0., 'PVInh_b': 0., 'PVInh_tauw':1e9,
    # --> Disinhibitory population (PVInh, recurrent inhibition)
    'SOMInh_Gl':10., 'SOMInh_Cm':200.,'SOMInh_Trefrac':3.,
    'SOMInh_El':-60., 'SOMInh_Vthre':-53., 'SOMInh_Vreset':-60., 'SOMInh_deltaV':0.,
    'SOMInh_a':0., 'SOMInh_b': 0., 'SOMInh_tauw':1e9,
}


if sys.argv[-1]=='plot':
    # ######################
    # ## ----- Plot ----- ##
    # ######################
    
    ## load file
    data = ntwk.load_dict_from_hdf5('sinusoid_input_data.h5')
    # ## plot
    fig, _ = ntwk.raster_and_Vm_plot(data, smooth_population_activity=10.)
    
    plt.show()
else:
    NTWK = ntwk.build_populations(Model, ['l23Exc', 'PVInh', 'SOMInh'],                                  
                                  AFFERENT_POPULATIONS=['l4Exc'],                                  
                                  with_raster=True,
                                  with_Vm=4,
                                  verbose=True)

    ntwk.build_up_recurrent_connections(NTWK, SEED=5, verbose=True)

    #######################################
    ########### AFFERENT INPUTS ###########
    #######################################

    faff = 2.

    t_array = ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']

    faff[t_array>1.] = 2.+3*np.sin(2*np.pi*5*(t_array[t_array>1.]-1))

    # # # afferent excitation onto cortical excitation and inhibition
    for i, tpop in enumerate(['l23Exc', 'PVInh', 'SOMInh']): # both on excitation and inhibition
        ntwk.construct_feedforward_input(NTWK, tpop, 'l4Exc',
                                         t_array, faff,
                                         verbose=True,
                                         SEED=int(37*faff+i)%37)

    ################################################################
    ## --------------- Initial Condition ------------------------ ##
    ################################################################
    ntwk.initialize_to_rest(NTWK)

    #####################
    ## ----- Run ----- ##
    #####################
    network_sim = ntwk.collect_and_run(NTWK, verbose=True)

    ntwk.write_as_hdf5(NTWK, filename='sinusoid_input_data.h5')
    print('Results of the simulation are stored as:', 'sinusoid_input_data.h5')
    print('--> Run \"python sinusoid_input.py plot\" to plot the results')
