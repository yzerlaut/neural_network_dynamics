"""
Demo file simulating the Vogels and Abbott 2005 network 
adapted from http://brian2.readthedocs.io/en/2.0b4/examples/COBAHH.html
"""
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
    'N_Exc':3200, 'N_Inh':800,
    # synaptic weights
    'Q_Exc_Exc':6., 'Q_Exc_Inh':6., 
    'Q_Inh_Exc':67., 'Q_Inh_Inh':67., 
    # synaptic time constants
    'Tse':5., 'Tsi':10.,
    # synaptic reversal potentials
    'Ee':0., 'Ei': -80.,
    # connectivity parameters
    'p_Exc_Exc':0.02, 'p_Exc_Inh':0.02, 
    'p_Inh_Exc':0.02, 'p_Inh_Inh':0.02, 
    # simulation parameters
    'dt':0.1, 'tstop': 100., 'SEED':3, # low by default, see later
    ## ---------------------------------------------------------------------------------
    # === cellular properties (based on AdExp), population by population ===
    # --> Excitatory population (Exc, recurrent excitation)
    'Exc_Gl':10., 'Exc_Cm':200.,'Exc_Trefrac':3.,
    'Exc_El':-60., 'Exc_Vthre':-50., 'Exc_Vreset':-60., 'Exc_delta_v':0.,
    'Exc_a':0., 'Exc_b': 0., 'Exc_tauw':1e9,
    # --> Inhibitory population (Inh, recurrent inhibition)
    'Inh_Gl':10., 'Inh_Cm':200.,'Inh_Trefrac':3.,
    'Inh_El':-60., 'Inh_Vthre':-50., 'Inh_Vreset':-60., 'Inh_delta_v':0.,
    'Inh_a':0., 'Inh_b': 0., 'Inh_tauw':1e9,
}

if sys.argv[-1]=='plot':
    ## load file
    data = ntwk.load_dict_from_hdf5('Vogels-Abbott.h5')
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
    ## we build and run the simulation
    NTWK = ntwk.build_populations(Model, ['Exc', 'Inh'],
                                  with_raster=True, with_Vm=4,
                                  # with_synaptic_currents=True,
                                  # with_synaptic_conductances=True,
                                  verbose=True)

    ntwk.build_up_recurrent_connections(NTWK, SEED=5, verbose=True)

    ################################################################
    ## --------------- Initial Condition ------------------------ ##
    ################################################################
    for i in range(2):
        NTWK['POPS'][i].V = (-65+5*np.random.randn(NTWK['POPS'][i].N))*ntwk.mV # random Vm

    # then excitation
    NTWK['POPS'][0].GExcExc = abs(40+15*np.random.randn(NTWK['POPS'][0].N))*ntwk.nS
    NTWK['POPS'][0].GInhExc = abs(200+120*np.random.randn(NTWK['POPS'][0].N))*ntwk.nS
    # # then inhibition
    NTWK['POPS'][1].GExcInh = abs(40+15*np.random.randn(NTWK['POPS'][1].N))*ntwk.nS
    NTWK['POPS'][1].GInhInh = abs(200+120*np.random.randn(NTWK['POPS'][1].N))*ntwk.nS

    # #####################
    # ## ----- Run ----- ##
    # #####################
    network_sim = ntwk.collect_and_run(NTWK, verbose=True)
    ntwk.write_as_hdf5(NTWK, filename='Vogels-Abbott.h5')

    print('Results of the simulation are stored as:', 'Vogels-Abbott.h5')
    print('--> Run \"python coba_LIF.py plot\" to plot the results')
    
    



