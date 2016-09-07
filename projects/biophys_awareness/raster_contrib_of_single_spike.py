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
    AFFERENCE_ARRAY = [{'Q':1., 'N':4000, 'pconn':0.05},
                       {'Q':1., 'N':1000, 'pconn':0.05}]
    rate_array = ramp_rise_then_constant(t_array, 0., 50., 0, args.f_ext)
    
    EXC_ACTS, INH_ACTS, SPK_TIMES, SPK_IDS = [], [], [], []

    for seed in range(1, args.nsim+1):

        M = get_connectivity_and_synapses_matrix('CONFIG1', number=len(NTWK))
        if args.Qe!=0:
            M[0,0]['Q'], M[0,1]['Q'] = args.Qe, args.Qe
        if args.Qi!=0:
            M[1,0]['Q'], M[1,1]['Q'] = args.Qi, args.Qi
            
        POPS, RASTER, POP_ACT = build_populations(NTWK, M, with_raster=True, with_pop_act=True)

        initialize_to_rest(POPS, NTWK) # (fully quiescent State as initial conditions)

        AFF_SPKS, AFF_SYNAPSES = construct_feedforward_input(POPS,
                                                             AFFERENCE_ARRAY,\
                                                             t_array,
                                                             rate_array,\
                                                             pop_for_conductance='A',
                                                             SEED=seed)
        SYNAPSES = build_up_recurrent_connections(POPS, M, SEED=seed+1)

        # Then single spike addition
        # spikes tergetting randomly one neuron in the network
        Nspikes = int((args.tstop-args.stim_start)/args.stim_delay)
        spike_times = args.stim_start+np.arange(Nspikes)*args.stim_delay+np.random.randn(Nspikes)*args.stim_jitter
        spike_ids = np.empty(0, dtype=int)
        Spk_times = np.empty(0, dtype=int)
        for ii in range(len(spike_times)):
            # non repetitive ids
            spike_ids = np.concatenate([spike_ids, np.random.choice(np.arange(POPS[0].N), args.duplicate_spikes, replace=False)])

        spike_times = np.sort(np.meshgrid(spike_times, np.ones(args.duplicate_spikes))[0].flatten())
        INPUT_SPIKES = brian2.SpikeGeneratorGroup(POPS[0].N, spike_ids, spike_times*brian2.ms) # targetting purely exc pop
        
        FEEDFORWARD = brian2.Synapses(INPUT_SPIKES, POPS[0], on_pre='GAA_post += w', model='w:siemens')
        FEEDFORWARD.connect('i==j')
        FEEDFORWARD.w=args.Qe_spike*brian2.nS

        EXC_SPIKES, INH_SPIKES = brian2.SpikeMonitor(POPS[0]), brian2.SpikeMonitor(POPS[1])
        
        net = brian2.Network(brian2.collect())
        # manually add the generated quantities
        net.add(POPS, SYNAPSES, RASTER, POP_ACT, AFF_SPKS, AFF_SYNAPSES, FEEDFORWARD, INPUT_SPIKES, EXC_SPIKES, INH_SPIKES) 
        net.run(args.tstop*brian2.ms)

        EXC_ACTS.append(POP_ACT[0].smooth_rate(window='flat',\
                                               width=args.smoothing*brian2.ms)/brian2.Hz)
        INH_ACTS.append(POP_ACT[1].smooth_rate(window='flat',\
                                               width=args.smoothing*brian2.ms)/brian2.Hz)
        SPK_TIMES.append(spike_times)
        SPK_IDS.append(spike_ids)

    np.savez(args.filename, args=args,
             exc_spk = np.array(EXC_SPIKES.t),
             inh_spk = np.array(INH_SPIKES.t),
             exc_ids = np.array(EXC_SPIKES.i),
             inh_ids = np.array(INH_SPIKES.i),
             plot=get_plotting_instructions())

def get_plotting_instructions():
    return """
fig, ax = plt.subplots(1, figsize=(4,3))
args = data['args'].all()
plt.plot(1e3*data['exc_spk'], data['exc_ids'], 'g.')
plt.plot(1e3*data['inh_spk'], data['exc_ids'].max()+data['inh_ids'], 'r.')
set_plot(ax, xlabel='time (ms)', ylabel='neuron number', ylim=[3200,4200], xlim=[100,200])
fig.savefig('/Users/yzerlaut/Desktop/fig.svg')
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
    parser.add_argument("--tstop",help="simulation duration (ms)",type=float, default=200.)
    parser.add_argument("--nsim",help="number of simulations (different seeds used)", type=int, default=1)
    parser.add_argument("--smoothing",help="smoothing window (flat) of the pop. act.",type=float, default=0.5)
    # network architecture
    parser.add_argument("--Ne",help="excitatory neuron number", type=int, default=4000)
    parser.add_argument("--Ni",help="inhibitory neuron number", type=int, default=1000)
    parser.add_argument("--Qe", help="weight of excitatory spike (0. means default)", type=float, default=0.)
    parser.add_argument("--Qi", help="weight of inhibitory spike (0. means default)", type=float, default=0.)
    parser.add_argument("--f_ext",help="external drive (Hz)",type=float, default=4.)
    # stimulation (single spike) properties
    parser.add_argument("--stim_start", help="time of the start for the additional spike (ms)", type=float, default=100.)
    parser.add_argument("--stim_delay",help="we multiply the single spike on the trial at this (ms)",type=float, default=50.)
    parser.add_argument("--stim_jitter",help="we jitter the spike times with a gaussian distrib (ms)",type=float, default=5.)
    parser.add_argument("--Qe_spike", help="weight of additional excitatory spike", type=float, default=1.)
    parser.add_argument("--duplicate_spikes", help="we duplicate the spike over neurons", type=int, default=1)
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
