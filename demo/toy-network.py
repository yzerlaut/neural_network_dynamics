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
    'N_Exc':5, 'N_Inh':2, 'N_AffExc':4, 'N_OptoInh':1,
    # synaptic weights
    'Q_Exc_Exc':16., 'Q_Exc_Inh':16., 
    'Q_Inh_Exc':40., 'Q_Inh_Inh':40., 
    'Q_AffExc_Exc':16., 'Q_AffExc_Inh':16.,
    'Q_OptoInh_Inh':200.,
    # synaptic time constants
    'Tsyn_Exc':5., 'Tsyn_Inh':5.,
    # synaptic reversal potentials
    'Erev_Exc':0., 'Erev_Inh': -80.,
    # connectivity parameters -- USELESS BUT NON ZERO TO ENABLE CONNECTION
    'p_Exc_Exc':1, 'p_Exc_Inh':1, 'p_Inh_Exc':1, 'p_Inh_Inh':1, 
    'p_AffExc_Exc':1, 'p_AffExc_Inh':1, 
    'p_OptoInh_Inh':1,
    # simulation parameters
    'dt':0.1, 'tstop': 1870., 
    'SEED':3, # connectivity seed
    ## ---------------------------------------------------------------------------------
    # === cellular properties (based on AdExp), population by population ===
    # --> Excitatory population (Exc, recurrent excitation)
    'Exc_Gl':10., 'Exc_Cm':200.,'Exc_Trefrac':3.,
    'Exc_El':-70., 'Exc_Vthre':-50., 'Exc_Vreset':-70., 'Exc_deltaV':0.,
    'Exc_a':0., 'Exc_b': 0., 'Exc_tauw':1e9,
    # --> Inhibitory population (Inh, recurrent inhibition)
    'Inh_Gl':10., 'Inh_Cm':200.,'Inh_Trefrac':3.,
    'Inh_El':-70., 'Inh_Vthre':-53., 'Inh_Vreset':-70., 'Inh_deltaV':0.,
    'Inh_a':0., 'Inh_b': 0., 'Inh_tauw':1e9,
}


####################################
########    RECURRENCE   ###########
####################################

NTWK = ntwk.build.populations(Model, ['Exc', 'Inh'],
                              AFFERENT_POPULATIONS=['AffExc', 'OptoInh'],
                              with_raster=True, 
                              with_Vm=7, 
                              verbose=True)

NTWK['REC_SYNAPSES'] = []

# excitatory - excitatory
Pairs = [\
        [0, 1],
        [0, 3],
        [0, 4],

        [1, 2],
        [1, 3],

        [2, 0],
        [2, 2],
         ]
NTWK['REC_SYNAPSES'].append(\
    ntwk.build.connections_from_pairs(NTWK, Pairs, 'Exc', 'Exc'))
# excitatory - inhibitory
Pairs = [\
        [3, 0],
        [4, 0],
        [2, 1],
        [4, 1],
         ]
NTWK['REC_SYNAPSES'].append(\
    ntwk.build.connections_from_pairs(NTWK, Pairs, 'Exc', 'Inh'))
# inhibitory - excitatory
Pairs = [\
        [0, 3],
        [0, 4],
        [1, 2],
         ]
NTWK['REC_SYNAPSES'].append(\
    ntwk.build.connections_from_pairs(NTWK, Pairs, 'Inh', 'Exc'))


if sys.argv[-1]=='stim':

    Model['tstop'] = 900.
    t0, jitter, Dt, delay = 200, 4, 150, 15
    # afferent to excitation 
    set_of_events = [[] for i in range(Model['N_Exc'])]

    # stim A
    for event in [0, 2*Dt, 3*Dt, 4*Dt]:
        set_of_events[0].append(t0+event)
        set_of_events[1].append(t0+event)
        set_of_events[2].append(t0+event)
    # stim B
    for event in [Dt, 2*Dt, 3*Dt+delay, 4*Dt+delay]:
        set_of_events[2].append(t0+event+3)
        set_of_events[3].append(t0+event)
        set_of_events[4].append(t0+event)

    ntwk.stim.events_one_synapse_per_neuron(NTWK, 'Exc', 'AffExc',
                                            set_of_events)


    set_of_events = [[] for i in range(Model['N_Inh'])]
    # stim A
    for event in [0, 2*Dt, 3*Dt, 4*Dt]:
        set_of_events[1].append(t0+event+3)
        set_of_events[1].append(t0+event+5)
    # stim B
    # for event in [Dt, 2*Dt, 3*Dt+delay, 4*Dt+delay]:
        # set_of_events[0].append(t0+event+3)

    ntwk.stim.events_one_synapse_per_neuron(NTWK, 'Inh', 'AffExc',
                                            set_of_events)

    # optogenetic silencing
    ntwk.stim.events_one_synapse_per_neuron(NTWK, 'Inh', 'OptoInh',
                                            [780+np.arange(230) for k in range(2)])
    sim = ntwk.collect_and_run(NTWK, 
                               verbose=True)

elif sys.argv[-1]=='steps':
    # current steps

    Model['tstop'] = 900.
    II = []

    i = 1
    for on, dur, amp, pop, Id in zip(\
                                     list(130+np.arange(7)*100),
                                     [25,20,25,20,25,25,25],
                                     [280, 280, 280, 280, 280, 280,280],
                                     [0,1,0,1,0,0,0], 
                                     [0,1,1,0,2,4,3]):
        # on
        exec("def update%i(NTWK): NTWK['POPS'][%i].I0[%i] = %f*ntwk.pA" % \
                (i, pop, Id, amp))
        exec("II.append({'time':%f, 'function':update%i})" % (on, i))
        i+=1
        # off
        exec("def update%i(NTWK): NTWK['POPS'][%i].I0[%i] = 0*ntwk.pA" % \
                (i, pop, Id))
        exec("II.append({'time':%f, 'function':update%i})" % (on+dur, i))
        i+=1

    ntwk.build.initialize_to_rest(NTWK)

    sim = ntwk.collect_and_run(NTWK, 
                               INTERMEDIATE_INSTRUCTIONS=II,
                               verbose=True)

## write data
ntwk.recording.write_as_hdf5(NTWK, filename='data/toy-network.h5')

## load file
data = ntwk.recording.load_dict_from_hdf5('data/toy-network.h5')

fig, AX = ntwk.plots.pretty(data, tzoom=[100, 1000] if sys.argv[-1]=='stim' else None,
                           axes_extents = dict(Raster=2, Vm=7),
                           Vm_args=dict(vpeak=10, shift=25),
                           Raster_args=dict(ms=4, with_annot=False),
          fig_args=dict(figsize=(2.2, 0.3), dpi=150,
                        hspace=0.3, bottom=0.2, top=0.2, 
                        left=0.1, right = 0.1),
                           COLORS=['tab:green', 'tab:red'])
plt.savefig('temp.svg')
plt.show()
