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
    'N_Exc':7, 'N_Inh':2, 'N_AffExc':4,
    # synaptic weights
    'Q_Exc_Exc':1., 'Q_Exc_Inh':1., 
    'Q_Inh_Exc':10., 'Q_Inh_Inh':10., 
    'Q_AffExc_Exc':3., 'Q_AffExc_Inh':3.,
    # synaptic time constants
    'Tsyn_Exc':5., 'Tsyn_Inh':5.,
    # synaptic reversal potentials
    'Erev_Exc':0., 'Erev_Inh': -80.,
    # connectivity parameters -- USELESS BUT NON ZERO TO ENABLE CONNECTION
    'p_Exc_Exc':1, 'p_Exc_Inh':1, 'p_Inh_Exc':1, 'p_Inh_Inh':1, 
    'p_AffExc_Exc':1, 'p_AffExc_Inh':1, 
    # simulation parameters
    'dt':0.1, 'tstop': 500., 
    'SEED':3, # connectivity seed
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


####################################
########    RECURRENCE   ###########
####################################

NTWK = ntwk.build.populations(Model, ['Exc', 'Inh'],
                              AFFERENT_POPULATIONS=['AffExc'],
                              with_raster=True, with_Vm=7, verbose=True)

Pairs = [\
        [0, 3],
        [1, 3],
        [1, 5],
        [2, 3],
                ]
        
NTWK['REC_SYNAPSES'] = []
NTWK['REC_SYNAPSES'].append(\
    ntwk.build.connections_from_pairs(NTWK, Pairs, 'Exc', 'Exc'))

#######################################
########    STIMULATION     ###########
#######################################

# afferent to excitation 
set_of_events = [[] for e in range(Model['N_Exc'])]
set_of_events[1].append(50) # ms
ntwk.stim.events_one_synapse_per_neuron(NTWK, 'Exc', 'AffExc',
                                        set_of_events)

# afferent to excitation 
set_of_events = [[] for e in range(Model['N_Inh'])]
set_of_events[0].append(20) # ms
ntwk.stim.events_one_synapse_per_neuron(NTWK, 'Inh', 'AffExc',
                                        set_of_events)



#####################
## ----- Run ----- ##
#####################

def update1(NTWK):
    NTWK['POPS'][0].I0[0] = 500*ntwk.pA
    NTWK['POPS'][0].I0[1] = 500*ntwk.pA
def update2(NTWK):
    NTWK['POPS'][0].I0[0] = 0*ntwk.pA
    NTWK['POPS'][1].I0[1] = 500*ntwk.pA


ntwk.build.initialize_to_rest(NTWK)

sim = ntwk.collect_and_run(NTWK, 
                           INTERMEDIATE_INSTRUCTIONS=[{'time':100, 'function':update1},
                                                      {'time':300, 'function':update2}],
                           verbose=True)



## write data
ntwk.recording.write_as_hdf5(NTWK, filename='data/toy-network.h5')

## load file
data = ntwk.recording.load_dict_from_hdf5('data/toy-network.h5')

fig, _ = ntwk.plots.raster_and_Vm(data, Vm_args=dict(vpeak=-10, shift=12))
plt.show()
