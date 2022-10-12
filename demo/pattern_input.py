import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pylab as plt
import ntwk

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
    'N_Exc':400, 'N_Inh':100, 'N_AffExcTV':50, 'N_AffExcBG':50, # DIVIDED BY 10 FOR TROUBLESHOOTING
    # synaptic weights
    'Q_Exc_Exc':1., 'Q_Exc_Inh':1., 
    'Q_Inh_Exc':10., 'Q_Inh_Inh':10., 
    'Q_AffExcTV_Exc':3., 'Q_AffExcTV_Inh':3., 
    'Q_AffExcBG_Exc':3., 'Q_AffExcBG_Inh':3., 
    # synaptic time constants
    'Tse':5., 'Tsi':5.,
    # synaptic reversal potentials
    'Ee':0., 'Ei': -80.,
    # connectivity parameters
    'p_Exc_Exc':0.02, 'p_Exc_Inh':0.02, 
    'p_Inh_Exc':0.02, 'p_Inh_Inh':0.02, 
    'p_AffExcBG_Exc':0.1, 'p_AffExcBG_Inh':0.1, 
    'p_AffExcTV_Exc':0.1, 'p_AffExcTV_Inh':0.1, 
    # simulation parameters
    'dt':0.1, 'tstop': 100., 
    'SEED':3, # connectivity seed
    'BG-SEED':3, # background activity seed
    'STIM-SEED':3, # stimulus seed
    ## ---------------------------------------------------------------------------------
    # === cellular properties (based on AdExp), population by population ===
    # --> Excitatory population (Exc, recurrent excitation)
    'Exc_Gl':10., 'Exc_Cm':200.,'Exc_Trefrac':3.,
    'Exc_El':-60., 'Exc_Vthre':-50., 'Exc_Vreset':-60., 'Exc_deltaV':0.,
    'Exc_a':0., 'Exc_b': 0., 'Exc_tauw':1e9,
    # --> Inhibitory population (Inh, recurrent inhibition)
    'Inh_Gl':10., 'Inh_Cm':200.,'Inh_Trefrac':3.,
    'Inh_El':-60., 'Inh_Vthre':-53., 'Inh_Vreset':-60., 'Inh_deltaV':0.,
    'Inh_a':0., 'Inh_b': 0., 'Inh_tauw':1e9,
}

StimPattern = {'indices':[], 'times':[]}
for event in range(3):
    StimPattern['times'] += list(30*(1+event*np.ones(50)))
    StimPattern['indices'] += list(range(50))

if sys.argv[-1]=='plot':

    ######################
    ## ----- Plot ----- ##
    ######################
    
    ## load file
    data = ntwk.recording.load_dict_from_hdf5('pattern_input_data.h5')

    # plot input patterm
    fig, ax = plt.subplots(1)
    ax.set_title('stim. pattern')
    ax.plot(data['AffExcTV_times'], data['AffExcTV_indices'], 'ko', ms=1)
    ax.set_xlim([0, Model['tstop']])
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('nrn ID')

    # ## plot
    fig, _ = ntwk.plots.raster_and_Vm_plot(data, smooth_population_activity=10.)
    
    plt.show()

else:

    #############################
    ## ----- Simulation ------ ##
    #############################

    REC_POPS = ['Exc', 'Inh']
    AFF_POPS = ['AffExcBG', 'AffExcTV']

    NTWK = ntwk.build.populations(Model, REC_POPS,
                                  AFFERENT_POPULATIONS=AFF_POPS,
                                  with_raster=True, 
                                  with_Vm=1,
                                  # with_synaptic_currents=True,
                                  # with_synaptic_conductances=True,
                                  verbose=True)

    ntwk.build.recurrent_connections(NTWK, SEED=5, verbose=True)

    #######################################
    ########### AFFERENT INPUTS ###########
    #######################################
    t_array = ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']

    # background activity
    faff = 1.
    # # # afferent excitation onto cortical excitation and inhibition
    for i, tpop in enumerate(REC_POPS): # both on excitation and inhibition
        ntwk.stim.construct_feedforward_input(NTWK, tpop, 'AffExcBG',
                                              t_array, faff+0.*t_array,
                                              verbose=True,
                                              SEED=int(Model['BG-SEED']))

    # build connectivity matrices for the stimulus
    ntwk.build.fixed_afference(NTWK, ['AffExcTV'], REC_POPS)

    # stimulus activity
    for i, tpop in enumerate(REC_POPS): # both on excitation and inhibition
        ntwk.stim.construct_feedforward_input(NTWK, tpop, 'AffExcTV',
                                              t_array, 0.*t_array, # no background aff
                                              additional_spikes_in_terms_of_pre_pop=StimPattern,
                                              verbose=True,
                                              SEED=int(Model['STIM-SEED']))

    ################################################################
    ## --------------- Initial Condition ------------------------ ##
    ################################################################
    ntwk.build.initialize_to_rest(NTWK)

    #####################
    ## ----- Run ----- ##
    #####################
    network_sim = ntwk.collect_and_run(NTWK, verbose=True)

    ######################
    ## ----- Write ---- ##
    ######################

    NTWK['AffExcTV_indices'] = StimPattern['indices']
    NTWK['AffExcTV_times'] = StimPattern['times']
    ntwk.recording.write_as_hdf5(NTWK, 
                                 ARRAY_KEYS=['AffExcTV_indices', 'AffExcTV_times'],
                                 filename='pattern_input_data.h5')

    print('Results of the simulation are stored as:', 'pattern_input_data.h5')
    print('--> Run \"python pattern_input_data.py plot\" to plot the results')
    
    
