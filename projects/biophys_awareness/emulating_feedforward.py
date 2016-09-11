"""
This script sets up an afferent inhomogenous Poisson process onto the populations
"""
import brian2, string
import numpy as np

import sys
sys.path.append('../../')
from ntwk_build.syn_and_connec_construct import build_populations,\
    build_up_recurrent_connections,\
    initialize_to_rest
from ntwk_build.syn_and_connec_library import get_connectivity_and_synapses_matrix
from ntwk_stim.waveform_library import double_gaussian, ramp_rise_then_constant
from ntwk_stim.connect_afferent_input import construct_feedforward_input
from common_libraries.data_analysis.array_funcs import find_coincident_duplicates_in_two_arrays


def run_sim(args):
    ### SIMULATION PARAMETERS
    brian2.defaultclock.dt = args.DT*brian2.ms
    t_array = np.arange(int(args.tstop/args.DT))*args.DT

    NTWK = [{'name':'exc', 'N':args.Ne, 'type':'AdExp'},
            {'name':'inh', 'N':args.Ni, 'type':'LIF'}]
    AFFERENCE_ARRAY = [{'Q':args.Qe_ff, 'N':args.Ne, 'pconn':args.pconn},
                       {'Q':args.Qe_ff, 'N':args.Ne, 'pconn':args.pconn}]
    
    M = get_connectivity_and_synapses_matrix('CONFIG1', number=len(NTWK), verbose=args.verbose)
    if args.Qe!=0:
        M[0,0]['Q'], M[0,1]['Q'] = args.Qe, args.Qe
    if args.Qi!=0:
        M[1,0]['Q'], M[1,1]['Q'] = args.Qi, args.Qi
        
    EXC_ACTS_ACTIVE1, EXC_ACTS_ACTIVE2, EXC_ACTS_ACTIVE3  = [], [], []
    EXC_ACTS_REST1, EXC_ACTS_REST2, EXC_ACTS_REST3  = [], [], []

    for EXC_ACTS1, f_ext in zip([EXC_ACTS_ACTIVE1, EXC_ACTS_REST1],
                                [args.fext, 0.]):
        rate_array = f_ext+double_gaussian(t_array, args.stim_start,\
                                           args.stim_T0, args.stim_T1, args.f_stim)
        for seed in range(1, args.nsim+1):

            ## SIMULATION 1
            print('[initializing simulation 1 ...], f_ext0=', f_ext, 'seed=', seed)
            POPS, RASTER, POP_ACT = build_populations(NTWK, M, with_raster=True, with_pop_act=True, verbose=args.verbose)
            initialize_to_rest(POPS, NTWK) # (fully quiescent State as initial conditions)
            AFF_SPKS, AFF_SYNAPSES = construct_feedforward_input(POPS, AFFERENCE_ARRAY,\
                                                                 t_array, rate_array, pop_for_conductance='A', SEED=seed)
            SYNAPSES = build_up_recurrent_connections(POPS, M, SEED=seed+1)
            net = brian2.Network(brian2.collect())
            # manually add the generated quantities
            net.add(POPS, SYNAPSES, RASTER, POP_ACT, AFF_SPKS, AFF_SYNAPSES) 
            print('[running simulation 1 ...]')
            net.run(args.tstop*brian2.ms)
            print('[simulation 1 done -> saving output]')
            EXC_ACTS1.append(POP_ACT[0].smooth_rate(window='flat',\
                                                    width=args.smoothing*brian2.ms)/brian2.Hz)
        
    for EXC_ACTS1, EXC_ACTS2, f_ext in zip([EXC_ACTS_ACTIVE1, EXC_ACTS_REST1],
                                           [EXC_ACTS_ACTIVE2,EXC_ACTS_REST2],
                                           [args.fext, 0.]):
        rate_array = np.array(EXC_ACTS1).mean(axis=0)
        rate_array = np.array([ee if ee<args.fext+2*args.f_stim else args.fext for ee in rate_array])
        for seed in range(1, args.nsim+1):
            ## SIMULATION 2
            print('[initializing simulation 2 ...], f_ext0=', f_ext, 'seed=', seed)

            POPS, RASTER, POP_ACT = build_populations(NTWK, M, with_raster=True, with_pop_act=True, verbose=args.verbose)
            initialize_to_rest(POPS, NTWK) # (fully quiescent State as initial conditions)
            ## OUTPUT AS INPUT !!!
            AFF_SPKS, AFF_SYNAPSES = construct_feedforward_input(POPS, AFFERENCE_ARRAY,\
                                                                 t_array, rate_array, pop_for_conductance='A', SEED=seed)
            SYNAPSES = build_up_recurrent_connections(POPS, M, SEED=seed+1)
            net = brian2.Network(brian2.collect())
            # manually add the generated quantities
            net.add(POPS, SYNAPSES, RASTER, POP_ACT, AFF_SPKS, AFF_SYNAPSES) 
            print('[running simulation 2 ...]')
            net.run(args.tstop*brian2.ms)
            print('[simulation 2 done -> saving output]')
            EXC_ACTS2.append(POP_ACT[0].smooth_rate(window='flat',\
                                                   width=args.smoothing*brian2.ms)/brian2.Hz)

    for EXC_ACTS2, EXC_ACTS3, f_ext in zip([EXC_ACTS_ACTIVE2, EXC_ACTS_REST2],
                                           [EXC_ACTS_ACTIVE3,EXC_ACTS_REST3],
                                           [args.fext, 0.]):
        rate_array = np.array(EXC_ACTS2).mean(axis=0)
        rate_array = np.array([ee if ee<args.fext+2*args.f_stim else args.fext for ee in rate_array])
        for seed in range(1, args.nsim+1):
            ## SIMULATION 3
            print('[initializing simulation 3 ...], f_ext0=', f_ext, 'seed=', seed)
            POPS, RASTER, POP_ACT = build_populations(NTWK, M, with_raster=True, with_pop_act=True, verbose=args.verbose)
            initialize_to_rest(POPS, NTWK) # (fully quiescent State as initial conditions)
            ## OUTPUT AS INPUT !!!
            rate_array = np.array([ee if ee<args.fext+2*args.f_stim else args.fext for ee in EXC_ACTS2[-1]])
            AFF_SPKS, AFF_SYNAPSES = construct_feedforward_input(POPS, AFFERENCE_ARRAY,\
                                                                 t_array, rate_array, pop_for_conductance='A', SEED=seed)
            SYNAPSES = build_up_recurrent_connections(POPS, M, SEED=seed+1)
            net = brian2.Network(brian2.collect())
            # manually add the generated quantities
            net.add(POPS, SYNAPSES, RASTER, POP_ACT, AFF_SPKS, AFF_SYNAPSES) 
            print('[running simulation 3 ...]')
            net.run(args.tstop*brian2.ms)
            print('[simulation 3 done -> saving output]')
            EXC_ACTS3.append(POP_ACT[0].smooth_rate(window='flat',\
                                                   width=args.smoothing*brian2.ms)/brian2.Hz)
            
    np.savez(args.filename, args=args,
             EXC_ACTS_ACTIVE1=np.array(EXC_ACTS_ACTIVE1), EXC_ACTS_ACTIVE2=np.array(EXC_ACTS_ACTIVE2), EXC_ACTS_ACTIVE3=np.array(EXC_ACTS_ACTIVE3),
             EXC_ACTS_REST1=np.array(EXC_ACTS_REST1), EXC_ACTS_REST2=np.array(EXC_ACTS_REST2), EXC_ACTS_REST3=np.array(EXC_ACTS_REST3),
             NTWK=NTWK, t_array=t_array,
             rate_array=rate_array, AFFERENCE_ARRAY=AFFERENCE_ARRAY,
             plot=get_plotting_instructions())

def get_plotting_instructions():
    return """
args = data['args'].all()
fig, AX = plt.subplots(2, figsize=(7,7))
plt.subplots_adjust(left=0.15, bottom=0.15, wspace=0.2, hspace=0.2)
active_resp, rest_resp = [], []
i0 = int((args.stim_start-2.*args.stim_T0)/args.DT)
i1 = min([int((args.stim_start+3.*args.stim_T1)/args.DT), len(data['t_array'])-10])
AX[1].plot(data['t_array'], data['EXC_ACTS_ACTIVE1'].mean(axis=0), 'r-', lw=1)
AX[1].plot(data['t_array'], data['EXC_ACTS_ACTIVE2'].mean(axis=0), 'k-', lw=1)
AX[1].plot(data['t_array'], data['EXC_ACTS_ACTIVE3'].mean(axis=0), 'k-', lw=1)
AX[1].plot(data['t_array'], data['EXC_ACTS_REST1'].mean(axis=0), 'r-', lw=1)
AX[1].plot(data['t_array'], data['EXC_ACTS_REST2'].mean(axis=0), 'k-', lw=1)
AX[1].plot(data['t_array'], data['EXC_ACTS_REST3'].mean(axis=0), 'k-', lw=1)
# for exc_act_active, exc_act_rest  in zip(data['EXC_ACTS_ACTIVE3'], data['EXC_ACTS_REST3']):
#     active_resp.append(exc_act_active[i0:i1].mean()-exc_act_active[i1:].mean())
#     rest_resp.append(exc_act_rest[i0:i1].mean()-exc_act_rest[i1:].mean())
#     AX[1].plot(data['t_array'], exc_act_rest, 'k-')
#     AX[1].plot(data['t_array'], exc_act_active, 'b-')
# for exc_act_active, exc_act_rest  in zip(data['EXC_ACTS_ACTIVE1'], data['EXC_ACTS_REST1']):
#     AX[1].plot(data['t_array'], exc_act_rest, 'k-')
#     AX[1].plot(data['t_array'], exc_act_active, 'b-')
# for exc_act_active, exc_act_rest  in zip(data['EXC_ACTS_ACTIVE2'], data['EXC_ACTS_REST2']):
#     AX[1].plot(data['t_array'], exc_act_rest, 'k-')
#     AX[1].plot(data['t_array'], exc_act_active, 'b-')
# AX[0].plot(f_ext, active_resp, 'b-')
# AX[0].plot(f_ext, rest_resp, 'k-')
# AX[0].plot(rest_resp, rest_resp, 'k--')
set_plot(AX[0], xlabel='drive freq. (Hz)', ylabel='mean exc. (Hz)')
set_plot(AX[1], xlabel='drive freq. (Hz)', ylabel='mean exc. (Hz)')
"""


if __name__=='__main__':
    import argparse
    # First a nice documentation 
    parser=argparse.ArgumentParser(description=
     """ 
     Investigates what is the network response of a single spike 
     """
    ,formatter_class=argparse.RawTextHelpFormatter)

    
    # simulation parameters
    parser.add_argument("--DT",help="simulation time step (ms)",type=float, default=0.1)
    parser.add_argument("--tstop",help="simulation duration (ms)",type=float, default=1000.)
    parser.add_argument("--nsim",help="number of simulations (different seeds used)", type=int, default=3)
    parser.add_argument("--smoothing",help="smoothing window (flat) of the pop. act.",type=float, default=4.9)
    # network architecture
    parser.add_argument("--Ne",help="excitatory neuron number", type=int, default=4000)
    parser.add_argument("--Ni",help="inhibitory neuron number", type=int, default=1000)
    parser.add_argument("--pconn", help="connection proba", type=float, default=0.05)
    parser.add_argument("--Qe", help="weight of excitatory spike (0. means default)", type=float, default=1.)
    parser.add_argument("--Qi", help="weight of inhibitory spike (0. means default)", type=float, default=4.)
    parser.add_argument("--Qe_ff", help="weight of excitatory spike FEEDFORWARD", type=float, default=2.5)
    parser.add_argument("--fext",help="baseline external drive (Hz)",type=float, default=8.5)
    parser.add_argument("--f_stim",help="stimulation (Hz)",type=float, default=2.)
    # stimulation (single spike) properties
    parser.add_argument("--stim_start", help="time of the start for the additional spike (ms)", type=float, default=800.)
    parser.add_argument("--stim_T0",help="we multiply the single spike on the trial at this (ms)",type=float, default=10.)
    parser.add_argument("--stim_T1",help="we multiply the single spike on the trial at this (ms)",type=float, default=20.)
    
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("-u", "--update_plot", help="plot the figures", action="store_true")
    parser.add_argument("--filename", '-f', help="filename",type=str, default='data.npz')
    args = parser.parse_args()

    if args.update_plot:
        data = dict(np.load(args.filename))
        data['plot'] = get_plotting_instructions()
        np.savez(args.filename, **data)
    else:
        run_sim(args)
