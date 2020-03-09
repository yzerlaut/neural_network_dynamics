import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pylab as plt
import main as ntwk


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
    'N_Exc':4000, 'N_Inh':1000, 'N_oscillExc':500, 'N_AffExc':500,
    # synaptic weights
    'Q_Exc_Exc':1., 'Q_Exc_Inh':1., 
    'Q_AffExc_Exc':3., 'Q_AffExc_Inh':3., 'Q_AffExc_oscillExc':3., 
    'Q_Inh_Exc':10., 'Q_Inh_Inh':10.,
    'Q_oscillExc_Inh':10., 
    # synaptic time constants
    'Tse':5., 'Tsi':5.,
    # synaptic reversal potentials
    'Ee':0., 'Ei': -80.,
    # connectivity parameters
    'p_Exc_Exc':0.05, 'p_Exc_Inh':0.05, 'p_Exc_oscillExc':0.05,
    'p_Inh_Exc':0.001, 'p_Inh_Inh':0.05, 'p_Inh_oscillExc':0.05,
    'p_oscillExc_Exc':0.1, 'p_oscillExc_Inh':0.05, 'p_oscillExc_oscillExc':0.05, 
    'p_AffExc_Exc':0.1, 'p_AffExc_Inh':0.1, 'p_AffExc_oscillExc':0.1, 
    # simulation parameters
    'dt':0.1, 'tstop': 1000., 'SEED':3, # low by default, see later
    ## ---------------------------------------------------------------------------------
    # === cellular properties (based on AdExp), population by population ===
    # --> Excitatory population (Exc, recurrent excitation)
    'Exc_Gl':10., 'Exc_Cm':200.,'Exc_Trefrac':3.,
    'Exc_El':-70., 'Exc_Vthre':-50., 'Exc_Vreset':-70., 'Exc_deltaV':0.,
    'Exc_a':0., 'Exc_b': 0., 'Exc_tauw':1e9,
    # --> Inhibitory population (Inh, recurrent inhibition)
    'Inh_Gl':10., 'Inh_Cm':200.,'Inh_Trefrac':3.,
    'Inh_El':-70., 'Inh_Vthre':-50., 'Inh_Vreset':-70., 'Inh_deltaV':0.,
    'Inh_a':0., 'Inh_b': 0., 'Inh_tauw':1e9,
    # --> Disinhibitory population (Inh, recurrent inhibition)
    'oscillExc_Gl':10., 'oscillExc_Cm':200.,'oscillExc_Trefrac':3.,
    'oscillExc_El':-70., 'oscillExc_Vthre':-50., 'oscillExc_Vreset':-70., 'oscillExc_deltaV':0.,
    'oscillExc_Ioscill_amp':20.*10, 'oscillExc_Ioscill_freq': 3.,
    'oscillExc_a':0., 'oscillExc_b': 0., 'oscillExc_tauw':1e9,
}



if sys.argv[-1]=='plot':
    # ######################
    # ## ----- Plot ----- ##
    # ######################
    
    ## load file
    data = ntwk.load_dict_from_hdf5('rhythmic_ntwk_data.h5')

    # ## plot
    fig, _ = ntwk.raster_and_Vm_plot(data, smooth_population_activity=10.)
    
    plt.show()

else:

    NTWK = ntwk.build_populations(Model, ['Exc', 'Inh', 'oscillExc'],
                                  AFFERENT_POPULATIONS=['AffExc'],
                                  with_raster=True,
                                  with_pop_act=True,
                                  with_Vm=4,
                                  # with_synaptic_currents=True,
                                  # with_synaptic_conductances=True,
                                  verbose=True)

    ntwk.build_up_recurrent_connections(NTWK, SEED=5, verbose=True)

    #######################################
    ########### AFFERENT INPUTS ###########
    #######################################

    faff = 0.5
    t_array = ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']
    # # # afferent excitation onto cortical excitation and inhibition
    for i, tpop in enumerate(['Exc', 'Inh', 'oscillExc']): # both on excitation and inhibition
        ntwk.construct_feedforward_input(NTWK, tpop, 'AffExc',
                                         t_array, faff+0.*t_array,
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

    ntwk.write_as_hdf5(NTWK, filename='rhythmic_ntwk_data.h5')
    print('Results of the simulation are stored as:', 'rhythmic_ntwk_data.h5')
    print('--> Run \"python rhythmic_ntwk.py plot\" to plot the results')
    
    
