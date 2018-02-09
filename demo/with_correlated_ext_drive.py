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
    'N_Exc':4000, 'N_Inh':1000, 'N_AffExc':500,
    # synaptic weights
    'Q_Exc_Exc':1., 'Q_Exc_Inh':1., 
    'Q_AffExc_Exc':3., 'Q_AffExc_Inh':3., 
    'Q_Inh_Exc':10., 'Q_Inh_Inh':10., 
    # synaptic time constants
    'Tse':5., 'Tsi':5.,
    # synaptic reversal potentials
    'Ee':0., 'Ei': -80.,
    # connectivity parameters
    'p_Exc_Exc':0.02, 'p_Exc_Inh':0.02, 
    'p_Inh_Exc':0.02, 'p_Inh_Inh':0.02, 
    'p_AffExc_Exc':0.1, 'p_AffExc_Inh':0.1, 
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
    'Inh_El':-60., 'Inh_Vthre':-53., 'Inh_Vreset':-60., 'Inh_delta_v':0.,
    'Inh_a':0., 'Inh_b': 0., 'Inh_tauw':1e9,
}


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

print('-------------------------------------------------------')
print('BROKEN SCRIPT, NEED TO UPDATE WRT TO NEW ntwk_stim')
print('-------------------------------------------------------')
# faff = 1.
# t_array = ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']
# # # # afferent excitation onto cortical excitation and inhibition
# for i, tpop in enumerate(['Exc', 'Inh']): # both on excitation and inhibition
#     ntwk.construct_feedforward_input_correlated(NTWK, tpop, 'AffExc',
#                                                 t_array, faff+0.*t_array,
#                                                 with_presynaptic_spikes=True,
#                                                 verbose=True,
#                                                 SEED=int(37*faff+i)%37)

################################################################
## --------------- Initial Condition ------------------------ ##
################################################################
ntwk.initialize_to_rest(NTWK)

#####################
## ----- Run ----- ##
#####################
network_sim = ntwk.collect_and_run(NTWK, verbose=True)

# ######################
# ## ----- Plot ----- ##
# ######################
ii=0
for pop in NTWK['RASTER']:
    plt.plot(pop.t/ntwk.ms, ii+pop.i, 'o')
    try:
        ii+=np.array(pop.i).max()
    except ValueError:
        print('No spikes')
ntwk.set_plot(plt.gca(), ['bottom'], xlabel='time (ms)', yticks=[])
ntwk.show()

for i in range(4):
    plt.plot(NTWK['VMS'][0][i].t/ntwk.ms, NTWK['VMS'][0][i].V/ntwk.mV)
ntwk.set_plot(plt.gca(), xlabel='time (ms)', ylabel='Vm (mV)')
ntwk.show()
