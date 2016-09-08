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
    
    EXC_ACTS, INH_ACTS, SPK_TIMES, SPK_IDS = [], [], [], []

    for f_ext, seed in zip(np.linspace(args.fext_min, args.fext_max, args.nsim),\
                           range(1, args.nsim+1)):

        print('[initializing simulation ...]')
        rate_array = f_ext+0.*t_array
        M = get_connectivity_and_synapses_matrix('CONFIG1', number=len(NTWK))
        if args.Qe!=0:
            M[0,0]['Q'], M[0,1]['Q'] = args.Qe, args.Qe
        if args.Qi!=0:
            M[1,0]['Q'], M[1,1]['Q'] = args.Qi, args.Qi
            
        POPS, RASTER, POP_ACT = build_populations(NTWK, M, with_raster=True,\
                                                  with_pop_act=True)

        initialize_to_rest(POPS, NTWK) # (fully quiescent State as initial conditions)

        AFF_SPKS, AFF_SYNAPSES = construct_feedforward_input(POPS,
                                                             AFFERENCE_ARRAY,\
                                                             t_array,
                                                             rate_array,\
                                                             pop_for_conductance='A',
                                                             SEED=seed)
        SYNAPSES = build_up_recurrent_connections(POPS, M, SEED=seed+1)

        net = brian2.Network(brian2.collect())
        # manually add the generated quantities
        net.add(POPS, SYNAPSES, RASTER, POP_ACT, AFF_SPKS, AFF_SYNAPSES) 
        print('[running simulation ...]')
        net.run(args.tstop*brian2.ms)
        print('[simulation done -> saving output]')

        EXC_ACTS.append(POP_ACT[0].smooth_rate(window='flat',\
                                             width=args.smoothing*brian2.ms)/brian2.Hz)
        INH_ACTS.append(POP_ACT[1].smooth_rate(window='flat',\
                                             width=args.smoothing*brian2.ms)/brian2.Hz)
    np.savez(args.filename, args=args, EXC_ACTS=np.array(EXC_ACTS),
             INH_ACTS=np.array(INH_ACTS), NTWK=NTWK, t_array=t_array,
             rate_array=rate_array, AFFERENCE_ARRAY=AFFERENCE_ARRAY,
             plot=get_plotting_instructions())

def get_plotting_instructions():
    return """
args = data['args'].all()
fig, AX = plt.subplots(2, figsize=(5,3))
plt.subplots_adjust(left=0.15, bottom=0.15, wspace=0.2, hspace=0.2)
f_ext = np.linspace(args.fext_min, args.fext_max, args.nsim)
mean_exc_freq = []
for exc_act in data['EXC_ACTS']:
    print(exc_act[int(args.tstop/2/args.DT)+1:].mean())
    mean_exc_freq.append(exc_act[int(args.tstop/2/args.DT)+1:].mean())
    AX[1].plot(exc_act)
AX[0].plot(f_ext, mean_exc_freq, 'k-')
AX[0].plot(mean_exc_freq, mean_exc_freq, 'k--')
set_plot(AX[0], xlabel='drive freq. (Hz)', ylabel='mean exc. (Hz)')
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
    parser.add_argument("--nsim",help="number of simulations (different seeds used)", type=int, default=3)
    parser.add_argument("--smoothing",help="smoothing window (flat) of the pop. act.",type=float, default=0.5)
    # network architecture
    parser.add_argument("--Ne",help="excitatory neuron number", type=int, default=4000)
    parser.add_argument("--Ni",help="inhibitory neuron number", type=int, default=1000)
    parser.add_argument("--pconn", help="connection proba", type=float, default=0.05)
    parser.add_argument("--Qe", help="weight of excitatory spike (0. means default)", type=float, default=0.)
    parser.add_argument("--Qi", help="weight of inhibitory spike (0. means default)", type=float, default=0.)
    parser.add_argument("--Qe_ff", help="weight of excitatory spike FEEDFORWARD", type=float, default=2.)
    parser.add_argument("--fext_min",help="min external drive (Hz)",type=float, default=0.5)
    parser.add_argument("--fext_max",help="min external drive (Hz)",type=float, default=10.)
    # stimulation (single spike) properties
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
