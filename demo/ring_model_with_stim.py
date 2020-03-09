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
    'N_Exc':8000, 'N_Inh':2000, 'N_AffExc':400,
    # synaptic weights
    'Q_Exc_Exc':1., 'Q_Exc_Inh':1., 
    'Q_AffExc_Exc':1., 'Q_AffExc_Inh':1., 
    'Q_Inh_Exc':5., 'Q_Inh_Inh':5., 
    # spatial decay
    'SpatialDecay_Exc_Exc':400, 'SpatialDecay_Exc_Inh':400, 
    'SpatialDecay_AffExc_Exc':40, 'SpatialDecay_AffExc_Inh':40, 
    'SpatialDecay_Inh_Exc':100, 'SpatialDecay_Inh_Inh':100, 
    # ms delay per neuron
    'Delay_Exc_Exc':10*2./4000, 'Delay_Exc_Inh':10*2./4000, # 10ms at the most distant neuron (4000/2.)
    'Delay_AffExc_Exc':10*2./400, 'Delay_AffExc_Inh':10*2./400, 
    'Delay_Inh_Exc':10*2./1000, 'Delay_Inh_Inh':10*2./1000, 
    # synaptic time constants
    'Tse':5., 'Tsi':5.,
    # synaptic reversal potentials
    'Ee':0., 'Ei': -80.,
    # connectivity parameters
    'p_Exc_Exc':0.05, 'p_Exc_Inh':0.05, 
    'p_Inh_Exc':0.05, 'p_Inh_Inh':0.05, 
    'p_AffExc_Exc':0.5, 'p_AffExc_Inh':0.5, 
    # simulation parameters
    'dt':0.1, 'tstop': 500., 'SEED':3, # low by default, see later
    ## ---------------------------------------------------------------------------------
    # === cellular properties (based on AdExp), population by population ===
    # --> Excitatory population (Exc, recurrent excitation)
    'Exc_Gl':10., 'Exc_Cm':200.,'Exc_Trefrac':5.,
    'Exc_El':-70., 'Exc_Vthre':-50., 'Exc_Vreset':-70., 'Exc_deltaV':2.,
    'Exc_a':4., 'Exc_b': 40., 'Exc_tauw':500,
    # --> Inhibitory population (Inh, recurrent inhibition)
    'Inh_Gl':10., 'Inh_Cm':200.,'Inh_Trefrac':5.,
    'Inh_El':-70., 'Inh_Vthre':-50., 'Inh_Vreset':-70., 'Inh_deltaV':0.5,
    'Inh_a':0., 'Inh_b': 0., 'Inh_tauw':1e9,
}

if sys.argv[-1]=='plot':
    # ######################
    # ## ----- Plot ----- ##
    # ######################
    
    ## load file
    data = ntwk.load_dict_from_hdf5('ring_ntwk_data.h5')

    # ## plot
    fig, _ = ntwk.raster_and_Vm_plot(data, smooth_population_activity=10.)
    
    plt.show()

else:

    NTWK = ntwk.build_populations(Model, ['Exc', 'Inh'],
                                  AFFERENT_POPULATIONS=['AffExc'],
                                  with_raster=True, with_Vm=4,
                                  # with_synaptic_currents=True,
                                  # with_synaptic_conductances=True,
                                  verbose=True)

    ntwk.build_up_recurrent_connections(NTWK, SEED=5, verbose=True, with_ring_geometry=True)

    #######################################
    ########### AFFERENT INPUTS ###########
    #######################################

    t_array = ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']
    faff = 3. + 0*t_array
    ntwk.construct_feedforward_input(NTWK, 'Inh', 'AffExc', t_array, faff)
    faff[(t_array>400) & (t_array>500)] += 2.
    ntwk.construct_feedforward_input(NTWK, 'Exc', 'AffExc', t_array, faff)
    

    ################################################################
    ## --------------- Initial Condition ------------------------ ##
    ################################################################
    ntwk.initialize_to_rest(NTWK)

    #####################
    ## ----- Run ----- ##
    #####################
    network_sim = ntwk.collect_and_run(NTWK, verbose=True)
    
    ntwk.write_as_hdf5(NTWK, filename='ring_ntwk_data.h5')
    print('Results of the simulation are stored as:', 'ring_ntwk_data.h5')
    print('--> Run \"python ring_ntwk.py plot\" to plot the results')
    
    
