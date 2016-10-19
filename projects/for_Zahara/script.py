"""
This script sets up an afferent inhomogenous Poisson process onto the populations
"""
import brian2, string
import numpy as np

import sys
sys.path.append('../../../')
sys.path.append('../../')
from ntwk_build.syn_and_connec_construct import build_populations,\
    build_up_recurrent_connections,\
    initialize_to_rest
from ntwk_build.syn_and_connec_library import get_connectivity_and_synapses_matrix
from ntwk_stim.waveform_library import double_gaussian, ramp_rise_then_constant
from ntwk_stim.connect_afferent_input import construct_feedforward_input
from common_libraries.data_analysis.array_funcs import find_coincident_duplicates_in_two_arrays


def run_sim(args, for_parameter_scan=False):
    ### SIMULATION PARAMETERS

    print('[initializing simulation ...]')
    brian2.defaultclock.dt = args.DT*brian2.ms
    t_array = np.arange(int(args.tstop/args.DT))*args.DT

    NTWK = [{'name':'exc', 'N':args.Ne, 'type':'AdExp'},
            {'name':'inh', 'N':args.Ni, 'type':'LIF'}]
    AFFERENCE_ARRAY = [{'Q':args.Qe_ff, 'N':args.Ne, 'pconn':args.pconn},
                       {'Q':args.Qe_ff, 'N':args.Ne, 'pconn':args.pconn}]
    rate_array = args.f_ext+0.*t_array
    
    EXC_ACTS, INH_ACTS, SPK_TIMES, SPK_IDS = [], [], [], []

    M = get_connectivity_and_synapses_matrix('CONFIG1', number=len(NTWK), verbose=args.verbose)
    if args.Qe!=0:
        M[0,0]['Q'], M[0,1]['Q'] = args.Qe, args.Qe
    if args.Qi!=0:
        M[1,0]['Q'], M[1,1]['Q'] = args.Qi, args.Qi

    POPS, RASTER, POP_ACT = build_populations(NTWK, M, with_raster=True, with_pop_act=True, verbose=args.verbose)

    initialize_to_rest(POPS, NTWK) # (fully quiescent State as initial conditions)

    AFF_SPKS, AFF_SYNAPSES = construct_feedforward_input(POPS,
                                                         AFFERENCE_ARRAY,\
                                                         t_array,
                                                         rate_array,\
                                                         pop_for_conductance='A',
                                                         SEED=args.SEED)
    SYNAPSES = build_up_recurrent_connections(POPS, M, SEED=args.SEED+1)

    EXC_SPIKES, INH_SPIKES = brian2.SpikeMonitor(POPS[0]), brian2.SpikeMonitor(POPS[1])

    net = brian2.Network(brian2.collect())
    # manually add the generated quantities
    
    if for_parameter_scan:
        trace_Ge_exc = brian2.StateMonitor(POPS[0], 'GAA', record=range(args.nrec))
        trace_Gi_exc = brian2.StateMonitor(POPS[0], 'GBA', record=range(args.nrec))
        trace_Vm_exc = brian2.StateMonitor(POPS[0], 'V', record=range(args.nrec))
        net.add(POPS, SYNAPSES, RASTER, POP_ACT, AFF_SPKS, AFF_SYNAPSES, EXC_SPIKES, INH_SPIKES, trace_Ge_exc, trace_Gi_exc, trace_Vm_exc) 
    else:
        net.add(POPS, SYNAPSES, RASTER, POP_ACT, AFF_SPKS, AFF_SYNAPSES, EXC_SPIKES, INH_SPIKES)
        
    print('[running simulation ...]')
    net.run(args.tstop*brian2.ms)
    print('[simulation done -> saving output]')

    EXC_ACTS = POP_ACT[0].smooth_rate(window='flat',\
                                           width=args.smoothing*brian2.ms)/brian2.Hz
    INH_ACTS = POP_ACT[1].smooth_rate(window='flat',\
                                           width=args.smoothing*brian2.ms)/brian2.Hz

    
    if for_parameter_scan:
        np.savez(args.filename, args=args,
                 mean_G_exc = np.array([np.array(x.GAA/brian2.nS).mean() for x in trace_Ge_exc]),
                 std_G_exc = np.array([np.array(x.GAA/brian2.nS).std() for x in trace_Ge_exc]),
                 mean_G_inh = np.array([np.array(x.GBA/brian2.nS).mean() for x in trace_Gi_exc]),
                 std_G_inh = np.array([np.array(x.GBA/brian2.nS).std() for x in trace_Gi_exc]),
                 mean_Vm = np.array([np.array(x.V/brian2.mV).mean() for x in trace_Vm_exc]),
                 std_Vm = np.array([np.array(x.V/brian2.mV).std() for x in trace_Vm_exc]))
    else:
        np.savez(args.filename, args=args,
             t_array=t_array,
             exc_act = np.array(EXC_ACTS),
             inh_act = np.array(INH_ACTS),
             exc_spk = np.array(EXC_SPIKES.t),
             inh_spk = np.array(INH_SPIKES.t),
             exc_ids = np.array(EXC_SPIKES.i),
             inh_ids = np.array(INH_SPIKES.i))


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
    parser.add_argument("--tstop",help="simulation duration (ms)",type=float, default=200.)
    parser.add_argument("--nsim",help="number of simulations (different seeds used)", type=int, default=1)
    parser.add_argument("--nrec",help="number of recorded neurons", type=int, default=4)
    parser.add_argument("--SEED",help="seed for numerical sims", type=int, default=3)
    parser.add_argument("--smoothing",help="smoothing window (flat) of the pop. act.",type=float, default=0.5)
    # network architecture
    parser.add_argument("--Ne",help="excitatory neuron number", type=int, default=4000)
    parser.add_argument("--Ni",help="inhibitory neuron number", type=int, default=1000)
    parser.add_argument("--pconn", help="connection proba", type=float, default=0.05)
    parser.add_argument("--Qe", help="weight of excitatory spike (0. means default)", type=float, default=0.)
    parser.add_argument("--Qi", help="weight of inhibitory spike (0. means default)", type=float, default=0.)
    parser.add_argument("--f_ext",help="external drive (Hz)", type=float, default=4.)
    parser.add_argument("--Qe_ff", help="weight of excitatory spike FEEDFORWARD", type=float, default=2.)
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("-p", "--plot", help="plot the figures", action="store_true")
    parser.add_argument("--params_scan", action="store_true")
    parser.add_argument("--filename", '-f', help="filename", type=str, default='data.npz')
    args = parser.parse_args()

    if args.plot:
        data = np.load('data.npz')
        args = data['args'].all()
        sys.path.append('../../../')
        import matplotlib.pylab as plt
        from common_libraries.graphs.ntwk_dyn_plot import RASTER_PLOT, POP_ACT_PLOT
        from common_libraries.graphs import my_graph
        RASTER_PLOT([1e3*data['exc_spk'],1e3*data['inh_spk']], [data['exc_ids'],data['inh_ids']])
        POP_ACT_PLOT(data['t_array'], [data['exc_act'],data['inh_act']])
        plt.show()
    else:
        run_sim(args, for_parameter_scan=args.params_scan)
