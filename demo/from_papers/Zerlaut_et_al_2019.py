import sys, pathlib, scipy
import numpy as np
import matplotlib.pylab as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
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
    'N_RecExc':4000, 'N_RecInh':1000, 'N_AffExc':100, 'N_DsInh':500,
    # synaptic weights (nS)
    'Q_RecExc_RecExc':2., 'Q_RecExc_RecInh':2., 
    'Q_RecInh_RecExc':10., 'Q_RecInh_RecInh':10., 
    'Q_AffExc_RecExc':4., 'Q_AffExc_RecInh':4., 
    'Q_AffExc_DsInh':4.,
    'Q_DsInh_RecInh':10., 
    # synaptic time constants (ms)
    'Tse':5., 'Tsi':5.,
    # synaptic reversal potentials (mV)
    'Ee':0., 'Ei': -80.,
    # connectivity parameters (proba.)
    'p_RecExc_RecExc':0.05, 'p_RecExc_RecInh':0.05, 
    'p_RecInh_RecExc':0.05, 'p_RecInh_RecInh':0.05, 
    'p_DsInh_RecInh':0.05, 
    'p_AffExc_RecExc':0.1, 'p_AffExc_RecInh':0.1, 
    'p_AffExc_DsInh':0.075,
    # afferent stimulation (Hz)
    'F_AffExc':10.,
    # simulation parameters (ms)
    'dt':0.1, 'tstop':3000, 'SEED':3, # low by default, see later
    ## ---------------------------------------------------------------------------------
    # === cellular properties (based on AdExp), population by population ===
    # --> Excitatory population (RecExc, recurrent excitation)
    'RecExc_Gl':10., 'RecExc_Cm':200.,'RecExc_Trefrac':5.,
    'RecExc_El':-70., 'RecExc_Vthre':-50., 'RecExc_Vreset':-70., 'RecExc_delta_v':0.,
    'RecExc_a':0., 'RecExc_b': 0., 'RecExc_tauw':1e9, 'RecExc_deltaV':0,
    # --> Inhibitory population (RecInh, recurrent inhibition)
    'RecInh_Gl':10., 'RecInh_Cm':200.,'RecInh_Trefrac':5.,
    'RecInh_El':-70., 'RecInh_Vthre':-53., 'RecInh_Vreset':-70., 'RecInh_delta_v':0.,
    'RecInh_a':0., 'RecInh_b': 0., 'RecInh_tauw':1e9, 'RecInh_deltaV':0,
    # --> Disinhibitory population (DsInh, disinhibition)
    'DsInh_Gl':10., 'DsInh_Cm':200.,'DsInh_Trefrac':5.,
    'DsInh_El':-70., 'DsInh_Vthre':-50., 'DsInh_Vreset':-70., 'DsInh_delta_v':0.,
    'DsInh_a':0., 'DsInh_b': 0., 'DsInh_tauw':1e9, 'DsInh_deltaV':0,
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
    data = ntwk.recording.load_dict_from_hdf5('Zerlaut_et_al_2019_data.h5')

    # ## plot
    fig, _ = ntwk.plots.raster_and_Vm_plot(data, smooth_population_activity=10.)
    
    plt.show()
else:
    
    NTWK = ntwk.build.populations(Model, ['RecExc', 'RecInh', 'DsInh'],
                                  AFFERENT_POPULATIONS=['AffExc'],
                                  with_raster=True, with_Vm=4,
                                  verbose=True)

    ntwk.build.recurrent_connections(NTWK, SEED=5, verbose=True)

    #######################################
    ########### AFFERENT INPUTS ###########
    #######################################

    
    t_array = ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']
    # # # afferent excitation onto thalamic excitation
    ntwk.stim.construct_feedforward_input(NTWK, 'RecExc', 'AffExc',
                                          t_array, waveform(t_array, Model),
                                          verbose=True, SEED=28)
    ntwk.stim.construct_feedforward_input(NTWK, 'DsInh', 'AffExc',
                                          t_array, waveform(t_array, Model),
                                          verbose=True, SEED=29)


    ################################################################
    ## --------------- Initial Condition ------------------------ ##
    ################################################################
    ntwk.build.initialize_to_rest(NTWK)

    #####################
    ## ----- Run ----- ##
    #####################
    ntwk.collect_and_run(NTWK, verbose=True)

    ntwk.recording.write_as_hdf5(NTWK, filename='Zerlaut_et_al_2019_data.h5')
    print('Results of the simulation are stored as:', '4pop_model_data.h5')
    print('--> Run \"python from_papers/Zerlaut_et-al_2019.py plot\" to plot the results')
