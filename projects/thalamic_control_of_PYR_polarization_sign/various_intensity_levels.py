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
            {'name':'inh', 'N':args.Ni, 'type':'LIF'}]
    AFFERENCE_ARRAY = [{'Q':args.Qe_thal, 'N':args.Ne, 'pconn':args.pconn},
                       {'Q':args.Qe_thal, 'N':args.Ne, 'pconn':args.pconn}]
    
    EXC_ACTS, INH_ACTS, MEAN_VM, STD_VM, EXC_RASTER, INH_RASTER = [], [], [], [], [], []

    INPUT_RATES = np.linspace(args.stim_min, args.stim_max, args.stim_discret)
    for f_ext in INPUT_RATES:

        rate_array = f_ext+0.*t_array
    
        M = get_connectivity_and_synapses_matrix('CONFIG1', number=len(NTWK))
        POPS, RASTER, POP_ACT = build_populations(NTWK, M, with_raster=True, with_pop_act=True)

        initialize_to_rest(POPS, NTWK) # (fully quiescent State as initial conditions)

        AFF_SPKS, AFF_SYNAPSES = construct_feedforward_input(POPS,
                                                             AFFERENCE_ARRAY,\
                                                             t_array,
                                                             rate_array,\
                                                             pop_for_conductance='A',
                                                             SEED=args.seed)
        SYNAPSES = build_up_recurrent_connections(POPS, M, SEED=args.seed+1)
        VM_MONITOR = brian2.StateMonitor(POPS[0], 'V', record=np.random.randint(POPS[0].N, size=args.recorded_neurons))

        net = brian2.Network(brian2.collect())
        # manually add the generated quantities
        net.add(POPS, SYNAPSES, RASTER, POP_ACT, AFF_SPKS, AFF_SYNAPSES, VM_MONITOR) 
        net.run(args.tstop*brian2.ms)

        exc_act = POP_ACT[0].smooth_rate(window='flat', width=args.smoothing*brian2.ms)/brian2.Hz
        inh_act = POP_ACT[1].smooth_rate(window='flat', width=args.smoothing*brian2.ms)/brian2.Hz
        EXC_ACTS.append(exc_act[int(args.trecord/args.DT):].mean())
        INH_ACTS.append(inh_act[int(args.trecord/args.DT):].mean())
        MEAN_VM.append(np.array(VM_MONITOR.V)[:,int(args.trecord/args.DT):].mean())
        STD_VM.append(np.array(VM_MONITOR.V)[:,int(args.trecord/args.DT):].std())
        
    np.savez(args.filename, args=args, NTWK=NTWK, INPUT_RATES=INPUT_RATES,
             EXC_ACTS=np.array(EXC_ACTS),INH_ACTS=np.array(INH_ACTS),
             MEAN_VM=np.array(MEAN_VM),STD_VM=np.array(STD_VM),
             plot=get_plotting_instructions())
    

def get_plotting_instructions():
    return """
args = data['args'].all()
fig, AX = plt.subplots(2, 1, figsize=(5,7))
data = np.load('data.npz')
print(data['INPUT_RATES'], data['EXC_ACTS'])
AX[0].plot(data['INPUT_RATES'], data['EXC_ACTS'], 'b')
AX[0].plot(data['INPUT_RATES'], data['INH_ACTS'], 'r')
set_plot(AX[0], xlabel='time (ms)', ylabel='pop. act. (Hz)')
AX[1].plot(data['INPUT_RATES'], data['MEAN_VM'], 'k')
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
    parser.add_argument("--trecord",help="simulation duration (ms)",type=float, default=400.)
    parser.add_argument("--nsim",help="number of simulations (different seeds used)", type=int, default=1)
    parser.add_argument("--recorded_neurons",help="number of recorded neurons for Vm", type=int, default=10)
    parser.add_argument("--seed",help="seed used", type=int, default=3)
    parser.add_argument("--smoothing",help="smoothing window (flat) of the pop. act.",type=float, default=0.5)
    # network architecture
    parser.add_argument("--Ne",help="excitatory neuron number", type=int, default=4000)
    parser.add_argument("--Ni",help="inhibitory neuron number", type=int, default=1000)
    parser.add_argument("--pconn", help="connection proba", type=float, default=0.05)
    # stimulation (single spike) properties
    parser.add_argument("--Qe_thal", help="thalamic excitatory weight (nS)", type=float, default=2.)
    parser.add_argument("--stim_min", help="min stim level (Hz)", type=float, default=0.)
    parser.add_argument("--stim_max",help="max stim level (Hz)",type=float, default=10.)
    parser.add_argument("--stim_discret",help="discretization of stim levels",type=int, default=3)
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
