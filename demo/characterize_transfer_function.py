import sys, pathlib
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
}


if sys.argv[-1]=='plot':
    # ######################
    # ## ----- Plot ----- ##
    # ######################
    data = np.load('tf_data.npy', allow_pickle=True).item()
    ntwk.plots.tf_2_variables(data,
                              xkey='F_Exc', ckey='F_Inh',
                              ylim=[1e-1, 100],
                              yticks=[0.1, 1, 10],
                              yticks_labels=['0.01', '0.1', '1', '10'],
                              ylabel='$\\nu_{out}$ (Hz)',
                              xticks=[0.1, 1, 10],
                              xticks_labels=['0.1', '1', '10'],
                              xlabel='$\\nu_{e}$ (Hz)')
    ntwk.show()
    
else:

    Model['filename'] = 'tf_data.npy'
    Model['NRN_KEY'] = 'Exc' # we scan this population
    Model['tstop'] = 10000
    Model['N_SEED'] = 3 # seed repetition
    Model['POP_STIM'] = ['Exc', 'Inh']
    Model['F_Exc_array'] = np.logspace(-1, 2, 40)
    Model['F_Inh_array'] = np.logspace(-1, 2, 10)
    ntwk.transfer_functions.generate(Model)
    print('Results of the simulation are stored as:', 'tf_data.npy')
    # print('--> Run \"python 3pop_model.py plot\" to plot the results')



    
