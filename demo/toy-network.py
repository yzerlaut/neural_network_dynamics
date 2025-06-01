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
    'Q_AffExc_Exc':3.,
    # synaptic time constants
    'Tsyn_Exc':5., 'Tsyn_Inh':5.,
    # synaptic reversal potentials
    'Erev_Exc':0., 'Erev_Inh': -80.,
    # connectivity parameters
    'p_Exc_Exc':0.02, 'p_Exc_Inh':0.02, 
    'p_Inh_Exc':0.02, 'p_Inh_Inh':0.02, 
    'p_AffExc_Exc':0.1, 
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

StimPattern = {'indices':[], 'times':[]}
for event in range(3):
    StimPattern['times'] += list(30*(1+event*np.ones(7)))
    StimPattern['indices'] += list(range(7))




REC_POPS = ['Exc', 'Inh']
AFF_POPS = ['AffExc']

NTWK = ntwk.build.populations(Model, REC_POPS,
                              AFFERENT_POPULATIONS=AFF_POPS,
                              with_raster=True, with_Vm=7, verbose=True)

M0 = np.array([\
        [ 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0] ])

ntwk.build.connections_from_matrices(NTWK, [[M0]])

#######################################
########### AFFERENT INPUTS ###########
#######################################
t_array = ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']

# background activity
faff = 1.
# # # afferent excitation onto cortical excitation and inhibition
for i, tpop in enumerate(['Exc']): # both on excitation and inhibition
    ntwk.stim.construct_feedforward_input(NTWK, tpop, 'AffExc',
                                          t_array, faff+0.*t_array,
                                          verbose=True)

"""
# build connectivity matrices for the stimulus
ntwk.build.fixed_afference(NTWK, ['AffExc'], REC_POPS)

# stimulus activity
for i, tpop in enumerate(['Exc']): # both on excitation and inhibition
    ntwk.stim.construct_feedforward_input(NTWK, tpop, 'AffExc',
                                          t_array, 0.*t_array, # no background aff
                                          additional_spikes_in_terms_of_pre_pop=StimPattern,
                                          verbose=True)

"""

################################################################
## --------------- Initial Condition ------------------------ ##
################################################################
ntwk.build.initialize_to_rest(NTWK)

#####################
## ----- Run ----- ##
#####################


def update1(NTWK):
    NTWK['POPS'][1].I0[0] = 200*ntwk.pA
def update2(NTWK):
    NTWK['POPS'][1].I0[0] = 0*ntwk.pA


network_sim = ntwk.collect_and_run(NTWK, 
                                   INTERMEDIATE_INSTRUCTIONS=[{'time':100, 'function':update1},
                                                              {'time':200, 'function':update2}],
                                   verbose=True)



######################
## ----- Write ---- ##
######################

NTWK['AffExc_indices'] = StimPattern['indices']
NTWK['AffExc_times'] = StimPattern['times']
ntwk.recording.write_as_hdf5(NTWK, 
                             # ARRAY_KEYS=['AffExc_indices', 'AffExc_times'],
                             filename='toy-network.h5')

## load file
data = ntwk.recording.load_dict_from_hdf5('toy-network.h5')

# plot input patterm
fig, ax = plt.subplots(1)
ax.set_title('stim. pattern')
# ax.plot(data['AffExcTV_times'], data['AffExcTV_indices'], 'ko', ms=1)
ax.set_xlim([0, Model['tstop']])
ax.set_xlabel('time (ms)')
ax.set_ylabel('nrn ID')

# ## plot
fig, _ = ntwk.plots.raster_and_Vm(data)

plt.show()
