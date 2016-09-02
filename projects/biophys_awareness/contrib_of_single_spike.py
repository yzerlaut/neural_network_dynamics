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
        
        net = brian2.Network(brian2.collect())
        # manually add the generated quantities
        net.add(POPS, SYNAPSES, RASTER, POP_ACT, AFF_SPKS, AFF_SYNAPSES, FEEDFORWARD, INPUT_SPIKES) 
        net.run(args.tstop*brian2.ms)

        EXC_ACTS.append(POP_ACT[0].smooth_rate(window='flat',\
                                               width=args.smoothing*brian2.ms)/brian2.Hz)
        INH_ACTS.append(POP_ACT[1].smooth_rate(window='flat',\
                                               width=args.smoothing*brian2.ms)/brian2.Hz)
        SPK_TIMES.append(spike_times)
        SPK_IDS.append(spike_ids)
        
    np.savez(args.filename, args=args, EXC_ACTS=np.array(EXC_ACTS),
             INH_ACTS=np.array(INH_ACTS), NTWK=NTWK, t_array=t_array,
             rate_array=rate_array, AFFERENCE_ARRAY=AFFERENCE_ARRAY,
             SPK_IDS=SPK_IDS, SPK_TIMES=SPK_TIMES,
             plot=get_plotting_instructions())

def get_plotting_instructions():
    return """
args = data['args'].all()
fig, AX = plt.subplots(2, 1, figsize=(5,7))
plt.subplots_adjust(left=0.15, bottom=0.15, wspace=0.2, hspace=0.2)
AX[0].plot(data['t_array'][-1000:], data['rate_array'][-1000:], 'b')
AX[0].plot(data['t_array'][-1000:], data['EXC_ACTS'].mean(axis=0)[-1000:], 'g')
AX[0].plot(data['t_array'][-1000:], data['INH_ACTS'].mean(axis=0)[-1000:], 'r')
t_zoom = np.linspace(-10, 30, int(40/args.DT)+1)
trace, counter = 0.*t_zoom, 0
for spike_times, exc_act in zip(data['SPK_TIMES'], data['EXC_ACTS']):
    i_plot = int(data['SPK_TIMES'].shape[0]*len(np.unique(spike_times))/20)
    for t_spk in np.unique(spike_times):
        i_spk = int(t_spk/args.DT)
        counter +=1
        trace += exc_act[i_spk+int(t_zoom[0]/args.DT):i_spk+int(t_zoom[-1]/args.DT)+1]
        if counter%i_plot==0:
            AX[1].plot(t_zoom, exc_act[i_spk+int(t_zoom[0]/args.DT):i_spk+int(t_zoom[-1]/args.DT)+1], '-', color='gray', lw=0.2)
AX[1].plot(t_zoom, trace/counter, 'k-', lw=2)
set_plot(AX[0], xlabel='time (ms)', ylabel='pop. act. (Hz)')
set_plot(AX[1], xlabel='time lag (ms)', ylabel='pop. act. (Hz)')
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
