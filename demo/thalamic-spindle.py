"""

"""
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
    ## UNIT SYSTEM is : ms, mV, pF, nS, pA, Hz (arbitrary and unconsistent, so see code)
    ## ---------------------------------------------------------------------------------
    # numbers of neurons in population
    'N_TcExc':100, 'N_ReExc':100, 
    'N_AffExc':100,
    # synaptic weights
    'Q_TcExc_ReExc':5., 
    'Q_ReExc_TcExc':5., 
    'Q_ReExc_ReExc':5., 
    'Q_AffExc_TcExc':3.,
    # synaptic time constants
    'Tsyn_Exc':5., 
    # synaptic reversal potentials
    'Erev_Exc':0., 
    # connectivity parameters
    'p_TcExc_ReExc':0.05, 
    'p_ReExc_ReExc':0.1, 
    'p_ReExc_TcExc':0.02, 
    'p_AffExc_TcExc':0.1, 
    # simulation parameters
    'dt':0.1, 'tstop': 300., 'SEED':3, # low by default, see later
    ## ---------------------------------------------------------------------------------
    # === cellular properties (based on AdExp), population by population ===
    # --> Thalamo-Cortical population (Exc)
    'TcExc_Gl':10., 'TcExc_Cm':200.,'TcExc_Trefrac':3.,
    'TcExc_El':-60., 'TcExc_Vthre':-50., 'TcExc_Vreset':-60., 'TcExc_deltaV':2.5,
    'TcExc_a':40., 'TcExc_b': 0., 'TcExc_tauw':600.,
    # --> Reticular population (Exc)
    'ReExc_Gl':10., 'ReExc_Cm':200.,'ReExc_Trefrac':3.,
    'ReExc_El':-60., 'ReExc_Vthre':-50., 'ReExc_Vreset':-60., 'ReExc_deltaV':2.5,
    'ReExc_a':80., 'ReExc_b': 0.05, 'ReExc_tauw':600.,
}


if sys.argv[-1]=='plot':
    # ######################
    # ## ----- Plot ----- ##
    # ######################
    
    ## load file
    data = ntwk.recording.load_dict_from_hdf5('thal-spindle.h5')

    # ## plot
    fig, _ = ntwk.plots.activity_plots(data, 
                                       pop_act_args=dict(smoothing=4,
                                                         subsampling=4))
    plt.show()
else:
    NTWK = ntwk.build.populations(Model, ['TcExc', 'ReExc'],
                                  AFFERENT_POPULATIONS=['AffExc'],
                                  with_raster=True,
                                  with_Vm=4,
                                  # with_synaptic_currents=True,
                                  # with_synaptic_conductances=True,
                                  verbose=True)

    ntwk.build.recurrent_connections(NTWK, SEED=5, verbose=True)

    #######################################
    ########### AFFERENT INPUTS ###########
    #######################################

    faff = 10.
    t_array = ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']
    # # # afferent excitation onto cortical excitation and inhibition
    for i, tpop in enumerate(['TcExc']): # both on excitation and inhibition
        ntwk.stim.construct_feedforward_input(NTWK, tpop, 'AffExc',
                                              t_array, faff+0.*t_array,
                                              verbose=True,
                                              SEED=int(37*faff+i)%37)

    ################################################################
    ## --------------- Initial Condition ------------------------ ##
    ################################################################
    ntwk.build.initialize_to_rest(NTWK)

    #####################
    ## ----- Run ----- ##
    #####################
    network_sim = ntwk.collect_and_run(NTWK, verbose=True)

    ntwk.recording.write_as_hdf5(NTWK, filename='thal-spindle.h5')
    print('Results of the simulation are stored as:', 'thal-spindle.h5')
    print('--> Run \"python thalamic-spindle.py plot\" to plot the results')
