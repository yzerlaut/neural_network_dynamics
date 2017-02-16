"""
This script sets up an afferent inhomogenous Poisson process onto the populations
"""
import brian2, string
import numpy as np

import sys, os
sys.path.append('../../')
sys.path.append(os.path.expanduser('~')+os.path.sep+'work')
from ntwk_build.syn_and_connec_construct import build_populations,\
    build_up_recurrent_connections,\
    initialize_to_rest
from ntwk_build.syn_and_connec_library import get_connectivity_and_synapses_matrix
from ntwk_stim.waveform_library import gaussian, ramp_rise_then_constant
from ntwk_stim.connect_afferent_input import construct_feedforward_input
from common_libraries.data_analysis.array_funcs import find_coincident_duplicates_in_two_arrays

def run_sim(args, return_firing_rate_only=False):
    
    """ SIMULATION PARAMETERS """

    brian2.defaultclock.dt = args.DT*brian2.ms
    t_array = np.arange(int(args.tstop/args.DT))*args.DT

    NTWK = [{'name':'exc', 'N':args.Ne, 'type':'AdExp'},
            {'name':'inh', 'N':args.Ni, 'type':'LIF'},
            {'name':'exc', 'N':args.Ne, 'type':'AdExp'},
            {'name':'inh', 'N':args.Ni, 'type':'LIF'},
            {'name':'exc', 'N':args.Ne, 'type':'AdExp'},
            {'name':'inh', 'N':args.Ni, 'type':'LIF'}]
    AFFERENCE_ARRAY = [{'Q':args.Qe_ff, 'N':args.Ne, 'pconn':args.pconn},
                       {'Q':args.Qe_ff, 'N':args.Ne, 'pconn':args.pconn}]
    
    M = get_connectivity_and_synapses_matrix('', number=len(NTWK))

    # Manually construct the 6 by 6 matrix:
    # recurrent excitation
    for key, val in zip(['Q', 'pconn', 'Erev', 'Tsyn'], [args.Qe, args.pconn, 0., 5.]):
        # recurrent exc-exc and exc-inh connection !
        for m in [M[0,0], M[2,2], M[4,4], M[0,1], M[2,3], M[4,5]]:
            m[key] = val
    # recurrent inhibition
    for key, val in zip(['Q', 'pconn', 'Erev', 'Tsyn'], [args.Qi, args.pconn, -80., 5.]):
        # recurrent inh.
        for m in [M[1,1], M[3,3], M[5,5], M[1,0], M[3,2], M[5,4]]:
            m[key] = val
    # feedforward excitation
    for key, val in zip(['Q', 'pconn', 'Erev', 'Tsyn'], [args.Qe_ff, args.pconn_ff, 0., 5.]):
        # feedforward excitatory connection on excitation and inhibition!
        for m in [M[0,2], M[2,4], M[0,3], M[2,5]]:
            m[key] = val
    
    # over various stims...
    EXC_ACTS_ACTIVE1, EXC_ACTS_ACTIVE2, EXC_ACTS_ACTIVE3  = [], [], []
    EXC_ACTS_REST1, EXC_ACTS_REST2, EXC_ACTS_REST3  = [], [], []

    for EXC_ACTS1, EXC_ACTS2, EXC_ACTS3, f_ext1, f_ext2, f_ext3 in zip([EXC_ACTS_ACTIVE1,EXC_ACTS_REST1],
                                                                      [EXC_ACTS_ACTIVE2,EXC_ACTS_REST2],
                                                                      [EXC_ACTS_ACTIVE3,EXC_ACTS_REST3],
                                                                      [args.fext1, 0.],
                                                                      [args.fext2, 0.],
                                                                      [args.fext3, 0.]):

        if (not return_firing_rate_only) or (f_ext1>0):
            for seed in range(1, args.nsim+1):

                print('[initializing simulation ...], f_ext0=', f_ext1, 'seed=', seed)

                # rising ramp for the external drive
                rate_array1 = f_ext1*np.array([tt/args.fext_rise if tt< args.fext_rise else 1 for tt in t_array])

                # now we add the repeated stimulation
                tt0 = args.fext_rise+args.stim_start
                while (tt0<args.tstop):
                    rate_array1+=gaussian(t_array, tt0,\
                                          args.stim_T0, args.f_stim)
                    tt0+=args.stim_periodicity

                POPS, RASTER, POP_ACT = build_populations(NTWK, M, with_raster=True,\
                                                          with_pop_act=True,
                                                          verbose=args.verbose)
                # (fully quiescent State as initial conditions)
                initialize_to_rest(POPS, NTWK)
                # afferent external drive on to each populations
                AFF_SPKS1,AFF_SYNAPSES1 = construct_feedforward_input(POPS[:2],
                                                                      AFFERENCE_ARRAY,\
                                                                      t_array,
                                                                      rate_array1,\
                                                                      pop_for_conductance='A',
                                                                      target_conductances=['A', 'B'],
                                                                      SEED=seed)
                rate_array2 = f_ext2*np.array([tt/args.fext_rise if tt< args.fext_rise else 1 for tt in t_array])
                AFF_SPKS2,AFF_SYNAPSES2 = construct_feedforward_input(POPS[2:4],
                                                                      AFFERENCE_ARRAY,\
                                                                      t_array,
                                                                      rate_array2,\
                                                                      pop_for_conductance='C',
                                                                      target_conductances=['C', 'D'],
                                                                      SEED=seed+15)
                rate_array3 = f_ext3*np.array([tt/args.fext_rise if tt< args.fext_rise else 1 for tt in t_array])
                AFF_SPKS3,AFF_SYNAPSES3 = construct_feedforward_input(POPS[4:6],
                                                                      AFFERENCE_ARRAY,\
                                                                      t_array,
                                                                      rate_array3,\
                                                                      pop_for_conductance='E',
                                                                      target_conductances=['E', 'F'],
                                                                      SEED=seed+37)

                SYNAPSES = build_up_recurrent_connections(POPS, M, SEED=seed+1)

                net = brian2.Network(brian2.collect())
                # manually add the generated quantities
                net.add(POPS, SYNAPSES, RASTER, POP_ACT,
                        AFF_SPKS1, AFF_SYNAPSES1, AFF_SPKS2, AFF_SYNAPSES2, AFF_SPKS3, AFF_SYNAPSES3) 
                print('[running simulation ...]')
                net.run(args.tstop*brian2.ms)
                print('[simulation done -> saving output]')

                EXC_ACTS1.append(POP_ACT[0].smooth_rate(window='flat',\
                                                               width=args.smoothing*brian2.ms)/brian2.Hz)
                EXC_ACTS2.append(POP_ACT[2].smooth_rate(window='flat',\
                                                       width=args.smoothing*brian2.ms)/brian2.Hz)
                EXC_ACTS3.append(POP_ACT[4].smooth_rate(window='flat',\
                                                       width=args.smoothing*brian2.ms)/brian2.Hz)

    if return_firing_rate_only:
        return EXC_ACTS_ACTIVE1[-1][-5000:].mean(), EXC_ACTS_ACTIVE2[-1][-5000:].mean(),\
            EXC_ACTS_ACTIVE3[-1][-5000:].mean()
    else:
        # save data
        np.savez(args.filename, args=args,
             EXC_ACTS_ACTIVE1=np.array(EXC_ACTS_ACTIVE1),
             EXC_ACTS_ACTIVE2=np.array(EXC_ACTS_ACTIVE2),
             EXC_ACTS_ACTIVE3=np.array(EXC_ACTS_ACTIVE3),
             EXC_ACTS_REST1=np.array(EXC_ACTS_REST1),
             EXC_ACTS_REST2=np.array(EXC_ACTS_REST2),
             EXC_ACTS_REST3=np.array(EXC_ACTS_REST3),
             NTWK=NTWK, t_array=t_array,
             rate_array1=rate_array1, AFFERENCE_ARRAY=AFFERENCE_ARRAY,
             plot=get_plotting_instructions())

        
def average_all_stim(ACTS, args):
    if len(ACTS.shape)>1:
        sim_average = ACTS.mean(axis=0) # averaging over simulations
    else:
        sim_average = ACTS
    dt = args.DT
    tt0 = args.fext_rise+args.stim_start
    n, n0 = int(8.*args.stim_T0/args.DT), int(3*args.stim_T0/args.DT)
    t, VV = (np.arange(n)-n0)*args.DT, []
    k = 0
    while (args.fext_rise+args.stim_start+k*args.stim_periodicity<args.tstop):
        ii = int((args.fext_rise+args.stim_start +k*args.stim_periodicity)/args.DT)
        VV.append(sim_average[ii-n0:ii-n0+n])
        k+=1
    return t, np.array(VV).mean(axis=0), np.array(VV).std(axis=0)


from scipy.optimize import minimize
def fit_a_gaussian(t, v, args):
    def gaussian_with_shift(X):
        t0, sT, amplitude, baseline = X
        return np.sum(np.abs(baseline+gaussian(t, t0, sT, amplitude)-v))
    x0 = [0, args.stim_T0/2., args.f_stim, 1.]
    res = minimize(gaussian_with_shift, x0, tol=1e-7, options={'maxiter':10000})
    onset, width, amp, baseline = res.x
    return onset, width, amp, baseline

def get_plotting_instructions():
    return """
args = data['args'].all()
fig, AX = plt.subplots(3, figsize=(6,6))
plt.subplots_adjust(left=0.25, bottom=0.05, wspace=0.2, hspace=0.2)
for i in range(3):
    AX[i].plot(data['EXC_ACTS_ACTIVE'+str(i+1)][0], 'k-')
    AX[i].plot(data['EXC_ACTS_REST'+str(i+1)][0], 'b-')
fig2, AX = plt.subplots(4, figsize=(3,6))
plt.subplots_adjust(left=0.25, bottom=0.05, wspace=0.2, hspace=0.2)
fig3, AX2 = plt.subplots(4, 3, figsize=(2,6))
plt.subplots_adjust(left=0.25, bottom=0.05, wspace=0.3, hspace=0.2)
from input_on_feedforward import *
try:
    # input
    t, v, sv = average_all_stim(data['rate_array1'], args)
    AX[0].plot(t, v, 'k-', lw=2)
    AX[0].plot([0,100], [0,0], lw=5)
    # set_plot(AX[0], ['left'], ylabel='rate (Hz)', xticks=[], yticks=[0, 1, 2, 3])
    _, width0, amp0, bsl = fit_a_gaussian(t, v, args)
    onset0 = t[v-bsl>.1][0]
    AX[0].arrow(onset0, 0.4, 0, -0.2, fc='k', ec='k', head_width=0.5, head_length=0.1, linewidth=2)

    # active state output
    onset_act, width_act, amp_act = [], [], []
    for ax, exc_act in zip(AX[1:], [data['EXC_ACTS_ACTIVE1'],
                                data['EXC_ACTS_ACTIVE2'],
                                data['EXC_ACTS_ACTIVE3']]):
        t, v, sv = average_all_stim(exc_act, args)
        ax.plot(t, v, 'b')
        ax.fill_between(t, v-sv, v+sv, color='b', alpha=.4)
        onset, width, amp, bsl = fit_a_gaussian(t, v, args)
        ax.plot(t, bsl+gaussian(t, onset, width, amp), 'r--')
        onset = t[v-bsl>sv[:100].mean()][0]
        ax.arrow(onset, bsl+3, 0, -2, fc='b', ec='b')
        ax.plot([onset], [bsl], 'kD')
        onset_act.append(onset0-onset)
        width_act.append(width)
        amp_act.append(amp)

    # resting state output
    onset_rest, width_rest, amp_rest = [], [], []
    for ax, exc_act in zip(AX[1:], [data['EXC_ACTS_REST1'],
                                data['EXC_ACTS_REST2'],
                                data['EXC_ACTS_REST3']]):
        t, v, sv = average_all_stim(exc_act, args)
        ax.plot(t, v, 'k')
        ax.fill_between(t, v-sv, v+sv, color='k', alpha=.3)
        # set_plot(ax, ['left'], xticks=[], ylabel='rate (Hz)', yticks=[0,5,10,15])
        onset, width, amp, bsl = fit_a_gaussian(t, v, args)
        ax.plot(t, bsl+gaussian(t, onset, width, amp), 'r--')
        onset = t[v>0][0]
        ax.arrow(onset, bsl+3, 0, -2, fc='k', ec='k')
        onset_rest.append(onset-onset0)
        width_rest.append(width)
        amp_rest.append(amp)
    for i, quant_act, quant_rest, ylabel, ylim in zip(range(3),
           [onset_act, width_act, amp_act], [onset_rest, width_rest, amp_rest],
           ['onset $t_0$ (ms)', 'width. T (ms)', 'amp. A (Hz)'], 
           [[0,30], [0,60], [0,10]]):
        for j in range(3):
           AX2[j+1, i].bar([0], quant_act[j])
           AX2[j+1, i].bar([1], quant_rest[j])
           AX2[j+1, i].plot([0.5, 0.5], ylim, 'w.')
           set_plot(AX2[j+1, i], ['left'], ylabel=ylabel, xticks=[], ylim=ylim) 
except ValueError:
    pass
"""

import argparse
# First a nice documentation 
parser=argparse.ArgumentParser(description=
 """ 
 Investigates what is the network response of a single spike 
 """
,formatter_class=argparse.RawTextHelpFormatter)


# simulation parameters
parser.add_argument("--DT",help="simulation time step (ms)",
                    type=float, default=0.1)
parser.add_argument("--tstop",help="simulation duration (ms)",
                    type=float, default=1500.)
parser.add_argument("--nsim",help="number of simulations (different seeds used)",
                    type=int, default=2)
parser.add_argument("--smoothing",help="smoothing window (flat) of the pop. act.",
                    type=float, default=0.5)
# network architecture
parser.add_argument("--Ne",help="excitatory neuron number",
                    type=int, default=4000)
parser.add_argument("--Ni",help="inhibitory neuron number",
                    type=int, default=1000)
parser.add_argument("--pconn", help="connection proba",
                    type=float, default=0.05)
parser.add_argument("--pconn_ff", help="connection proba",
                    type=float, default=0.02)
parser.add_argument("--Qe", help="weight of excitatory spike (0. means default)",
                    type=float, default=1.)
parser.add_argument("--Qi", help="weight of inhibitory spike (0. means default)",
                    type=float, default=4.)
parser.add_argument("--Qe_ff", help="weight of excitatory spike FEEDFORWARD",
                    type=float, default=2.5)
# external drive properties
parser.add_argument("--fext1",help="baseline external drive on layer 1 (Hz)",
                    type=float, default=4.)
parser.add_argument("--fext2",help="baseline external drive on layer 2 (Hz)",
                    type=float, default=3.)
parser.add_argument("--fext3",help="baseline external drive on layer 3 (Hz)",
                    type=float, default=3.)
parser.add_argument("--fext_rise",help="rise of external drive (ms)",
                    type=float, default=1000)
# stimulation (single spike) properties
parser.add_argument("--f_stim",help="peak external input (Hz)",
                    type=float, default=3.)
parser.add_argument("--stim_start",
                    help="time of the start for the additional spike after ext rise !! (ms)",
                    type=float, default=200.)
parser.add_argument("--stim_periodicity",
                    help="each xx ms, we send a new input (ms)",
                    type=float, default=600.)
parser.add_argument("--stim_T0",
                    help="we multiply the single spike on the trial at this (ms)",
                    type=float, default=60.)
# various settings
parser.add_argument("-v", "--verbose",
                    help="increase output verbosity", action="store_true")
parser.add_argument("-u", "--update_plot",
                    help="plot the figures", action="store_true")
parser.add_argument("--filename", '-f', help="filename",
                    type=str, default='data.npz')

if __name__=='__main__':
    
    args = parser.parse_args()
    if args.update_plot:
        data = dict(np.load(args.filename))
        data['plot'] = get_plotting_instructions()
        np.savez(args.filename, **data)
    else:
        run_sim(args)
