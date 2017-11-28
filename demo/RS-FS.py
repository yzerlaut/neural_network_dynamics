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
    'N_Exc':4000, 'N_Inh':1000, 'N_AffExc':400,
    # synaptic weights
    'Q_Exc_Exc':1., 'Q_Exc_Inh':1., 
    'Q_AffExc_Exc':1., 'Q_AffExc_Inh':1., 
    'Q_Inh_Exc':4., 'Q_Inh_Inh':4., 
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
    'Exc_El':-70., 'Exc_Vthre':-50., 'Exc_Vreset':-70., 'Exc_delta_v':2.,
    'Exc_a':0., 'Exc_b': 40., 'Exc_tauw':500,
    # --> Inhibitory population (Inh, recurrent inhibition)
    'Inh_Gl':10., 'Inh_Cm':200.,'Inh_Trefrac':5.,
    'Inh_El':-70., 'Inh_Vthre':-50., 'Inh_Vreset':-70., 'Inh_delta_v':0.5,
    'Inh_a':0., 'Inh_b': 0., 'Inh_tauw':1e9,
}

if sys.argv[-1]=='plot':
    ## load file
    data = ntwk.load_dict_from_hdf5('RS-FS.h5')
    print('excitatory firing activity: ', 1e3*len(data['iRASTER_Exc'])/data['tstop']/data['N_Exc'])
    print('inhibitory firing activity: ', 1e3*len(data['iRASTER_Inh'])/data['tstop']/data['N_Inh'])
    ## plot
    fig, AX = plt.subplots(2)
    AX[0].plot(data['tRASTER_Exc'], data['iRASTER_Exc'], 'bo', ms=2)
    AX[0].plot(data['tRASTER_Inh'], -data['iRASTER_Inh'], 'ro', ms=2)
    ntwk.set_plot(AX[0], [], xticks=[], yticks=[])
    for v in data['VMS_Exc']:
        AX[1].plot(np.arange(len(v))*data['dt'], v, 'k-', lw=1)
    ntwk.set_plot(AX[1], xlabel='time (ms)', ylabel='Vm (mV)')
    ntwk.show()

else:

    NTWK = ntwk.build_populations(Model, ['Exc', 'Inh'],
                                  AFFERENT_POPULATIONS=['AffExc'],
                                  with_raster=True, with_Vm=4,
                                  # with_synaptic_currents=True,
                                  # with_synaptic_conductances=True,
                                  verbose=True)

    ntwk.build_up_recurrent_connections(NTWK, SEED=5, verbose=True)

    #######################################
    ########### AFFERENT INPUTS ###########
    #######################################

    faff = 4.
    t_array = ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']
    # # # afferent excitation onto cortical excitation and inhibition
    for i, tpop in enumerate(['Exc', 'Inh']): # both on excitation and inhibition
        ntwk.construct_feedforward_input(NTWK, tpop, 'AffExc',
                                         t_array, faff+0.*t_array,
                                         with_presynaptic_spikes=True,
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
    ntwk.write_as_hdf5(NTWK, filename='RS-FS.h5')

    print('Results of the simulation are stored as:', 'RS-FS.h5')
    print('--> Run \"python RS-FS.py plot\" to plot the results')
    
