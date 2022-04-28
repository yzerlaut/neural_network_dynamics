import sys, pathlib, scipy
import numpy as np
import matplotlib.pylab as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
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
    'N_PyrExc':4000, 'N_PvInh':600, 'N_SstInh':400, 'N_VipInh':400, 'N_AffExc':100,
    # synaptic weights (nS)
    'Q_PyrExc_PyrExc':2., 'Q_PyrExc_PvInh':2., 'Q_PyrExc_SstInh':2., 
    'Q_PvInh_PyrExc':10., 'Q_PvInh_PvInh':10., 'Q_PvInh_SstInh':10., 
    'Q_SstInh_PyrExc':10., 'Q_SstInh_PvInh':10., 'Q_SstInh_SstInh':10.,  
    'Q_AffExc_PyrExc':4., 'Q_AffExc_PvInh':4., 'Q_AffExc_SstInh':4., 'Q_AffExc_VipInh':4.,
    'Q_SstInh_PvInh':10., 
    # synaptic time constants (ms)
    'Tse':5., 'Tsi':5.,
    # synaptic reversal potentials (mV)
    'Ee':0., 'Ei': -80.,
    # connectivity parameters (proba.)
    'p_PyrExc_PyrExc':0.05, 'p_PyrExc_PvInh':0.05, 'p_PyrExc_SstInh':0.05, 
    'p_PvInh_PyrExc':0.05, 'p_PvInh_PvInh':0.05, 'p_PvInh_SstInh':0.05, 
    'p_SstInh_PyrExc':0.05, 'p_SstInh_PvInh':0.05, 'p_SstInh_SstInh':0.05, 
    'p_VipInh_SstInh':0.125, 
    'p_AffExc_PyrExc':0.1, 'p_AffExc_PvInh':0.1, 'p_AffExc_SstInh':0.1, 
    'p_AffExc_VipInh':0.1,
    # afferent stimulation (Hz)
    'F_AffExc':10.,
    # simulation parameters (ms)
    'dt':0.1, 'tstop':3500, 'SEED':3, # low by default, see later
    ## ---------------------------------------------------------------------------------
    # === cellular properties (based on AdExp), population by population ===
    # --> Excitatory population (PyrExc, recurrent excitation)
    'PyrExc_Gl':10., 'PyrExc_Cm':200.,'PyrExc_Trefrac':5.,
    'PyrExc_El':-70., 'PyrExc_Vthre':-50., 'PyrExc_Vreset':-70., 'PyrExc_delta_v':0.,
    'PyrExc_a':0., 'PyrExc_b': 0., 'PyrExc_tauw':1e9, 'PyrExc_deltaV':0,
    # --> PV Inhibitory population
    'PvInh_Gl':10., 'PvInh_Cm':200.,'PvInh_Trefrac':5.,
    'PvInh_El':-70., 'PvInh_Vthre':-53., 'PvInh_Vreset':-70., 'PvInh_delta_v':0.,
    'PvInh_a':0., 'PvInh_b': 0., 'PvInh_tauw':1e9, 'PvInh_deltaV':0,
    # --> SST Inhibitory population
    'SstInh_Gl':10., 'SstInh_Cm':200.,'SstInh_Trefrac':5.,
    'SstInh_El':-70., 'SstInh_Vthre':-53., 'SstInh_Vreset':-70., 'SstInh_delta_v':0.,
    'SstInh_a':0., 'SstInh_b': 0., 'SstInh_tauw':1e9, 'SstInh_deltaV':0,
    # --> Disinhibitory population (VipInh, disinhibition)
    'VipInh_Gl':10., 'VipInh_Cm':200.,'VipInh_Trefrac':5.,
    'VipInh_El':-70., 'VipInh_Vthre':-50., 'VipInh_Vreset':-70., 'VipInh_delta_v':0.,
    'VipInh_a':0., 'VipInh_b': 0., 'VipInh_tauw':1e9, 'VipInh_deltaV':0,
    ## ---------------------------------------------------------------------------------
    # === afferent population waveform:
    'Faff1':4.,'Faff2':20.,'Faff3':8.,
    'DT':900., 'rise':50.
}


def waveform(t, Model):
    waveform = 0*t
    # first waveform
    for tt, fa in zip(\
         2.*Model['rise']+np.arange(3)*(3.*Model['rise']+Model['DT']),
                      [Model['Faff1'], Model['Faff2'], Model['Faff3']]):
        waveform += fa*\
             (1+scipy.special.erf((t-tt)/Model['rise']))*\
             (1+scipy.special.erf(-(t-tt-Model['DT'])/Model['rise']))/4
    return waveform



if sys.argv[-1]=='plot':
    # ######################
    # ## ----- Plot ----- ##
    # ######################
    
    ## load file
    data = ntwk.recording.load_dict_from_hdf5('canonical_data.h5')

    # ## plot
    fig, _ = ntwk.plots.raster_and_Vm_plot(data, smooth_population_activity=10.)
    
    plt.show()
else:
    
    NTWK = ntwk.build.populations(Model, ['PyrExc', 'PvInh', 'SstInh', 'VipInh'],
                                  AFFERENT_POPULATIONS=['AffExc'],
                                  with_raster=True, with_Vm=4,
                                  verbose=True)

    ntwk.build.recurrent_connections(NTWK, SEED=5,
                                     verbose=True)

    #######################################
    ########### AFFERENT INPUTS ###########
    #######################################
    t_array = ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']
    # # # afferent excitation onto thalamic excitation
    for iseed, pop in enumerate(['PyrExc', 'PvInh', 'SstInh', 'VipInh']):
        ntwk.stim.construct_feedforward_input(NTWK, pop, 'AffExc',
                                              t_array, waveform(t_array, Model),
                                              verbose=True, SEED=28+iseed)

    
    ################################################################
    ## --------------- Initial Condition ------------------------ ##
    ################################################################
    ntwk.build.initialize_to_rest(NTWK)

    #####################
    ## ----- Run ----- ##
    #####################
    ntwk.collect_and_run(NTWK, verbose=True)

    ntwk.recording.write_as_hdf5(NTWK, filename='canonical_data.h5')
    print('Results of the simulation are stored as:', 'canonical_data.h5')
    print('--> Run \"python from_papers/Canonical-Inh-circuit.py plot\" to plot the results')
