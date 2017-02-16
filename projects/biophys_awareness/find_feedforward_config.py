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
from ntwk_stim.waveform_library import double_gaussian, ramp_rise_then_constant
from ntwk_stim.connect_afferent_input import construct_feedforward_input
from common_libraries.data_analysis.array_funcs import find_coincident_duplicates_in_two_arrays

from input_on_feedforward import run_sim

def find_equal_activity_levels(args):
    desF = args.desired_freq
    i = 0 
    fe1, fe2, fe3 = run_sim(args, return_firing_rate_only=True)
    while (abs(fe1-desF)>0.5) and (abs(fe2-desF)>0.5) and (abs(fe3-desF)>0.5) and (i<100):
        print('step ', i, 'fe1=', fe1, 'fe2=', fe2, 'fe3=', fe3)
        if fe1>desF+0.5:
            args.fext1 -= 0.1
        elif fe1<desF-0.5:
            args.fext1 += 0.1
        elif fe2>desF+0.5:
            args.fext2 -= 0.1
        elif fe2<desF-0.5:
            args.fext2 += 0.1
        elif fe3>desF+0.5:
            args.fext3 -= 0.1
        elif fe3<desF-0.5:
            args.fext3 += 0.1
        print('===========>', 'f1=', args.fext1, 'f2=', args.fext2, 'f3=', args.fext3)
        fe1, fe2, fe3 = run_sim(args, return_firing_rate_only=True)
        i += 1

if __name__=='__main__':
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
                        type=float, default=1300.)
    parser.add_argument("--nsim",help="number of simulations (different seeds used)",
                        type=int, default=1)
    parser.add_argument("--smoothing",help="smoothing window (flat) of the pop. act.",
                        type=float, default=0.5)
    # network architecture
    parser.add_argument("--Ne",help="excitatory neuron number",
                        type=int, default=4000)
    parser.add_argument("--Ni",help="inhibitory neuron number",
                        type=int, default=1000)
    parser.add_argument("--pconn", help="connection proba",
                        type=float, default=0.05)
    parser.add_argument("--Qe", help="weight of excitatory spike (0. means default)",
                        type=float, default=1.)
    parser.add_argument("--Qi", help="weight of inhibitory spike (0. means default)",
                        type=float, default=4.)
    parser.add_argument("--Qe_ff", help="weight of excitatory spike FEEDFORWARD",
                        type=float, default=2.5)
    # external drive properties
    parser.add_argument("--desired_freq",help="desired frequency (Hz)",
                        type=float, default=3.)
    parser.add_argument("--fext1",help="baseline external drive on layer 1 (Hz)",
                        type=float, default=2.0)
    parser.add_argument("--fext2",help="baseline external drive on layer 2 (Hz)",
                        type=float, default=0.1)
    parser.add_argument("--fext3",help="baseline external drive on layer 3 (Hz)",
                        type=float, default=0.1)
    parser.add_argument("--fext_rise",help="rise of external drive (ms)",
                        type=float, default=500)
    # stimulation (single spike) properties
    parser.add_argument("--f_stim",help="peak external input (Hz)",
                        type=float, default=0.01)
    parser.add_argument("--stim_start",
                        help="time of the start for the additional spike after ext rise !! (ms)",
                        type=float, default=4000.)
    parser.add_argument("--stim_periodicity",
                        help="each xx ms, we send a new input (ms)",
                        type=float, default=400.)
    parser.add_argument("--stim_T0",
                        help="we multiply the single spike on the trial at this (ms)",
                        type=float, default=10.)
    parser.add_argument("--stim_T1",
                        help="we multiply the single spike on the trial at this (ms)",
                        type=float, default=40.)
    # various settings
    parser.add_argument("-v", "--verbose",
                        help="increase output verbosity", action="store_true")
    parser.add_argument("-u", "--update_plot",
                        help="plot the figures", action="store_true")
    parser.add_argument("--filename", '-f', help="filename",
                        type=str, default='data.npz')
    args = parser.parse_args()

    find_equal_activity_levels(args)
