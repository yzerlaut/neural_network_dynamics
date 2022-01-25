import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

import main as ntwk

from analyz.processing.signanalysis import gaussian_smoothing as smooth
from datavyz.main import graph_env




################################################################
## ------ Construct populations with their equations -------- ##
## ------------- with recurrent connections ----------------- ##
################################################################

REC_POPS = ['pyrExc', 'oscillExc', 'recInh', 'dsInh']
AFF_POPS = ['AffExc', 'NoiseExc']

# adding the same LIF props to all recurrent pops
LIF_props = {'Gl':10., 'Cm':200.,'Trefrac':5.,
             'El':-70, 'Vthre':-50., 'Vreset':-70., 'deltaV':0.,
             'a':0., 'b': 0., 'tauw':1e9}

Model = {
    ## -----------------------------------------
    ## UNIT SYSTEM is : ms, mV, pF, nS, pA, Hz
    ## (arbitrary and unconsistent, so see code)
    ## ------------------------------------------
    # numbers of neurons in population
    'N_pyrExc':4000, 'N_recInh':1000, 'N_dsInh':500, 'N_AffExc':100, 'N_NoiseExc':200, 'N_oscillExc':100,
    # synaptic time constants
    'Tse':5., 'Tsi':5.,
    # synaptic reversal potentials
    'Ee':0., 'Ei': -80.,
    # afferent stimulation
    'F_AffExc':10., 'F_NoiseExc':1.,
    # simulation parameters
    'dt':0.1, 'tstop': 4000., 'SEED':3, # low by default, see later
}

for pop in REC_POPS:
    for key, val in LIF_props.items():
        Model['%s_%s' % (pop, key)] = val
# adding the oscillatory feature to the oscillExc pop
Model['oscillExc_Ioscill_freq']=3.
Model['oscillExc_Ioscill_amp']= 10.*15.

# === adding synaptic weights ===
Qe, Qi = 2., 10 # nS
# loop oover the two population types
for aff in ['pyrExc', 'oscillExc', 'NoiseExc', 'AffExc']:
    for target in REC_POPS:
        Model['Q_%s_%s' % (aff, target)] = Qe
for aff in ['recInh', 'dsInh']:
    for target in REC_POPS:
        Model['Q_%s_%s' % (aff, target)] = Qi

# === initializing connectivity === #         
for aff in REC_POPS+AFF_POPS:
    for target in REC_POPS:
        Model['p_%s_%s' % (aff, target)] = 0.

# -------------------------------
# --- connectivity parameters ---
# -------------------------------
# ==> pyrExc
Model['p_pyrExc_pyrExc'] = 0.01
Model['p_pyrExc_oscillExc'] = 0.01
Model['p_pyrExc_recInh'] = 0.02
# ==> oscillExc
Model['p_oscillExc_oscillExc'] = 0.01
Model['p_oscillExc_pyrExc'] = 0.01
# ==> recInh
Model['p_recInh_recInh'] = 0.05
Model['p_recInh_pyrExc'] = 0.05
Model['p_recInh_oscillExc'] = 0.1
# ==> dsInh
Model['p_dsInh_recInh'] = 0.1
# ==> AffExc
Model['p_AffExc_dsInh'] = 0.2
Model['p_AffExc_recInh'] = 0.2
Model['p_AffExc_pyrExc'] = 0.05
# ==> NoiseExc
Model['p_NoiseExc_recInh'] = 0.1
Model['p_NoiseExc_pyrExc'] = 0.02
Model['p_NoiseExc_dsInh'] = 0.02
Model['p_NoiseExc_oscillExc'] = 0.2



if sys.argv[-1]=='plot':

    ######################
    ## ----- Plot ----- ##
    ######################

    ## load file
    data = ntwk.load_dict_from_hdf5('mean_field_data.h5')
    
    # ## plot
    fig, _ = ntwk.activity_plots(data,
                                 smooth_population_activity=10)
    
    ntwk.show()

elif sys.argv[-1]=='mf':

    data = ntwk.load_dict_from_hdf5('mean_field_data.h5')
    
    tstop, dt = 1e-3*data['tstop'], 1e-2
    t = np.arange(int(tstop/dt))*dt
    
    DYN_SYSTEM, INPUTS = {}, {}
    for rec in REC_POPS:
        Model['COEFFS_%s' % rec] = np.load('../configs/Network_Modulation_2020/COEFFS_pyrExc.npy')
        DYN_SYSTEM[rec] = {'aff_pops':['AffExc', 'NoiseExc'], 'x0':1e-2}
        INPUTS['AffExc_%s' % rec] = smooth(np.array([4.*int(tt) for tt in t]), int(.2/dt))
        INPUTS['NoiseExc_%s' % rec] = 3+0*t

    CURRENT_INPUTS = {'oscillExc':Model['oscillExc_Ioscill_amp']*(1-np.cos(Model['oscillExc_Ioscill_freq']*2*np.pi*t))/2.}
        
    X = ntwk.mean_field.solve_mean_field_first_order(Model,
                                                     DYN_SYSTEM,
                                                     INPUTS=INPUTS,
                                                     CURRENT_INPUTS=CURRENT_INPUTS,
                                                     dt=dt, tstop=tstop)

    ge = graph_env()
    fig, [ax0,ax1,ax2] = ge.figure(axes=(3,1), figsize=(3.,1.))

    ge.plot(t, Y=[INPUTS['AffExc_pyrExc'], INPUTS['NoiseExc_pyrExc']], COLORS=['k', ge.brown], ax=ax0)
    ge.plot(t, Y=[X['pyrExc']+1e-2, X['oscillExc'], X['recInh']+1e-2, X['dsInh']+1e-2],
            COLORS=[ge.g, ge.b, ge.r, ge.purple], ax=ax1, axes_args=dict(yticks=[0.1,1.,10], ylim=[.9e-2, 200], yscale='log'))
    ge.plot(Y=[smooth(x, 100) for x in [data['POP_ACT_pyrExc']+1e-2, data['POP_ACT_oscillExc'], data['POP_ACT_recInh']+1e-2, data['POP_ACT_dsInh']+1e-2]], 
            COLORS=[ge.g, ge.b, ge.r, ge.purple], ax=ax2, axes_args=dict(yticks=[0.1,1.,10], ylim=[.9e-2, 200], yscale='log'))
    ge.show()

else:

    ######################
    ## ----- Run  ----- ##
    ######################
    
    t_array = ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']
    faff = smooth(np.array([4*int(tt/1000) for tt in t_array]), int(200/0.1))
    
    fnoise = 3.

    #######################################
    ########### BUILD POPS ################
    #######################################
    
    NTWK = ntwk.build_populations(Model, REC_POPS,
                                  AFFERENT_POPULATIONS=AFF_POPS,
                                  with_pop_act=True,
                                  with_raster=True,
                                  with_Vm=4,
                                  verbose=True)

    ntwk.build_up_recurrent_connections(NTWK, SEED=5, verbose=True)

    #######################################
    ########### AFFERENT INPUTS ###########
    #######################################

    #  time-dep afferent excitation
    for i, tpop in enumerate(REC_POPS): # both on excitation and inhibition
        ntwk.construct_feedforward_input(NTWK, tpop, 'AffExc',
                                         t_array, faff,
                                         verbose=True,
                                         SEED=4)

    # # noise excitation
    for i, tpop in enumerate(REC_POPS): # both on excitation and inhibition
        ntwk.construct_feedforward_input(NTWK, tpop, 'NoiseExc',
                                         t_array, fnoise+0.*t_array,
                                         verbose=True,
                                         SEED=5)

    ################################################################
    ## --------------- Initial Condition ------------------------ ##
    ################################################################
    ntwk.initialize_to_rest(NTWK)

    #####################
    ## ----- Run ----- ##
    #####################
    network_sim = ntwk.collect_and_run(NTWK, verbose=True)

    #####################
    ## ----- Save ----- ##
    #####################
    ntwk.write_as_hdf5(NTWK, filename='mean_field_data.h5')
    print('Results of the simulation are stored as:', 'mean_field_data.h5')
    print('--> Run \"python mean_field.py plot\" to plot the results')
    print('--> Run \"python mean_field.py mf\" to run the associated MF and plot the results')

