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



def run_sim(args):
    ### SIMULATION PARAMETERS

    brian2.defaultclock.dt = args.DT*brian2.ms
    t_array = np.arange(int(args.tstop/args.DT))*args.DT

    NTWK = [{'name':'exc', 'N':args.Ne, 'type':'AdExp'},
            {'name':'inh', 'N':args.Ni, 'type':'EIF'}]
    AFFERENCE_ARRAY = [{'Q':1., 'N':400, 'pconn':0.1},
                       {'Q':1., 'N':400, 'pconn':0.1}]
    rate_array = ramp_rise_then_constant(t_array, 0., 30., 0., args.f_ext)
    
    EXC_ACTS, INH_ACTS = [], []

    for seed in range(1, args.nsim+1):

        M = get_connectivity_and_synapses_matrix('CONFIG1', number=len(NTWK))
        POPS, RASTER, POP_ACT = build_populations(NTWK, M, with_raster=True, with_pop_act=True)

        initialize_to_rest(POPS, NTWK) # (fully quiescent State as initial conditions)

        AFF_SPKS, AFF_SYNAPSES = construct_feedforward_input(POPS,
                                                             AFFERENCE_ARRAY,\
                                                             t_array,
                                                             rate_array,\
                                                             pop_for_conductance='A',
                                                             SEED=seed)
        SYNAPSES = build_up_recurrent_connections(POPS, M, SEED=seed)

        # Then single spike addition
        # input_spike = brian2.SpikeGeneratorGroup(POPS[0].N, [0], [args.spiketime])
        # fdfrwd_to_inh = Synapses(input_inh, inh_neurons, pre='Gei_post += w',\
        #                          model='w:siemens', connect='i==j')
        # fdfrwd_to_inh.w=P[0,1]['Q']*nS
        
        net = brian2.Network(brian2.collect())
        # manually add the generated quantities
        net.add(POPS, SYNAPSES, RASTER, POP_ACT, AFF_SPKS, AFF_SYNAPSES) 
        net.run(args.tstop*brian2.ms)

        EXC_ACTS.append(POP_ACT[0].smooth_rate(window='flat',\
                                       width=args.smoothing*brian2.ms)/brian2.Hz)
        INH_ACTS.append(POP_ACT[1].smooth_rate(window='flat',\
                                       width=args.smoothing*brian2.ms)/brian2.Hz)
        
    plot_data="fig = plt.figure(figsize=(5,3.5));data = np.load('data.npz');plt.plot(data['t_array'], data['rate_array'], 'b');plt.plot(data['t_array'], data['EXC_ACTS'].mean(axis=0), 'g');plt.plot(data['t_array'], data['INH_ACTS'].mean(axis=0), 'r')"
    
    np.savez('data.npz', args=args, EXC_ACTS=np.array(EXC_ACTS),
             INH_ACTS=np.array(INH_ACTS), NTWK=NTWK, t_array=t_array,
             rate_array=rate_array, AFFERENCE_ARRAY=AFFERENCE_ARRAY,
             plot=plot_data)

if __name__=='__main__':
    import argparse
    # First a nice documentation 
    parser=argparse.ArgumentParser(description=
     """ 
     Investigates what is the network response of a single spike 
     """
    ,formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--nsim",help="number of simulations (different seeds used)",\
                        type=int, default=5)
    parser.add_argument("--smoothing",help="smoothing window (flat) of the pop. act.",\
                        type=float, default=0.5)
    parser.add_argument("--DT",help="simulation time step (ms)",\
                        type=float, default=0.1)
    parser.add_argument("--tstop",help="simulation duration (ms)",\
                        type=float, default=100.)
    parser.add_argument("--f_ext",help="external drive (Hz)",type=float, default=10.)
    parser.add_argument("--stim_start",\
                        help="time of the start for the additional spike (ms)",\
                        type=float, default=50.)
    parser.add_argument("--stim_delay",\
                        help="we multiply the single spike on the trial at this (ms)",\
                        type=float, default=50.)
    parser.add_argument("--Ne",help="excitatory neuron number", type=int, default=4000)
    parser.add_argument("--Ni",help="inhibitory neuron number", type=int, default=1000)
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")
    parser.add_argument("-p", "--plot", help="plot the figures",
                        action="store_true")
    args = parser.parse_args()

    run_sim(args)
