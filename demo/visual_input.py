import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pylab as plt
import main as ntwk

from vision.earlyVis_model import earlyVis_model
from vision.stimuli import visual_stimulus
from vision.plots import plot as vision_plot


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
    'N_Exc':4000, 'N_Inh':1000, 'N_DsInh':500, 'N_AffExc':500,
    # synaptic weights
    'Q_Exc_Exc':1., 'Q_Exc_Inh':1., 
    'Q_AffExc_Exc':3., 'Q_AffExc_Inh':3., 'Q_AffExc_DsInh':3., 
    'Q_Inh_Exc':10., 'Q_Inh_Inh':10.,
    'Q_DsInh_Inh':10., 
    # synaptic time constants
    'Tse':5., 'Tsi':5.,
    # synaptic reversal potentials
    'Ee':0., 'Ei': -80.,
    # connectivity parameters
    'p_Exc_Exc':0.02, 'p_Exc_Inh':0.02, 
    'p_Inh_Exc':0.02, 'p_Inh_Inh':0.02, 
    'p_DsInh_Inh':0.02, 
    'p_AffExc_Exc':0.1, 'p_AffExc_Inh':0.1, 'p_AffExc_DsInh':0.1, 
    # simulation parameters
    'dt':0.1, 'tstop': 1000., 'SEED':3, # low by default, see later
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
    # --> Disinhibitory population (Inh, recurrent inhibition)
    'DsInh_Gl':10., 'DsInh_Cm':200.,'DsInh_Trefrac':3.,
    'DsInh_El':-60., 'DsInh_Vthre':-53., 'DsInh_Vreset':-60., 'DsInh_deltaV':0.,
    'DsInh_a':0., 'DsInh_b': 0., 'DsInh_tauw':1e9,
}


if sys.argv[-1]=='plot':
    # ######################
    # ## ----- Plot ----- ##
    # ######################

    vision_model = ntwk.earlyVis_model()
    ## load file
    # data = ntwk.load_dict_from_hdf5('visual_input_data.h5')

    # # ## plot
    # fig, _ = ntwk.raster_and_Vm_plot(data, smooth_population_activity=10.)
    
    # plt.show()
else:

    if False:
        
        params = ntwk.vision_params.copy()
        params['rf_theta'] = [3*np.pi/12., 5.*np.pi/12.]
        params['rf_psi'] = [0., 0.]
        params['rf_freq'] = [0.06, 0.08]
        params['Ncells'] = 50
        
        vision_model = ntwk.earlyVis_model(params=params)
        vision_model.full_process('drifting-grating', '')
        vision_model.save_data('drifting-grating-data.npz')
        
    else:
        
        vision_model = ntwk.earlyVis_model(from_file='drifting-grating-data.npz')

    ps = ntwk.vision_plot(model=vision_model)
    fig = ps.protocol_plot()

    ntwk.show()
    
    
    """
    NTWK = ntwk.build_populations(Model, ['Exc', 'Inh', 'DsInh'],
                                  AFFERENT_POPULATIONS=['AffExc'],
                                  with_raster=True,

                                  with_Vm=4,
                                  # with_synaptic_currents=True,
                                  # with_synaptic_conductances=True,
                                  verbose=True)

    ntwk.build_up_recurrent_connections(NTWK, SEED=5, verbose=True)

    #######################################
    ########### AFFERENT INPUTS ###########
    #######################################

    faff = 1.
    t_array = ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']
    # # # afferent excitation onto cortical excitation and inhibition
    for i, tpop in enumerate(['Exc', 'Inh', 'DsInh']): # both on excitation and inhibition
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

    ntwk.write_as_hdf5(NTWK, filename='visual_input_data.h5')
    print('Results of the simulation are stored as:', 'visual_input_data.h5')
    print('--> Run \"python visual_input.py plot\" to plot the results')
    """
