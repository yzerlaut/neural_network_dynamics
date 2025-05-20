"""
pulse protocol simulation for a single neuron model
"""

from .cell_library import *
from .cell_construct import *

from utils import plot_tools as pt

def current_pulse_sim(args, 
                      params=None, verbose=False):
    
    if params is None:
        params = get_neuron_params(args['NRN'])

    neurons, eqs = get_membrane_equation(params, [],\
                                         return_equations=True)

    if verbose:
        print(eqs)

    # V value initialization
    neurons.V = params['El']*brian2.mV
    trace = brian2.StateMonitor(neurons, 'V', record=0)
    spikes = brian2.SpikeMonitor(neurons)

    # rest run
    brian2.run(args['delay'] * brian2.ms)

    # 

    if len(args['durations'])==len(args['amplitudes']):
        durations = args['durations']
    else:
        durations = args['durations'][0]*np.ones(len(args['amplitudes']))

    I, ilast = np.zeros(len(trace.t)), len(trace.t)
    # loop over pulses
    for amp, dur in zip(args['amplitudes'], durations):
        # start
        neurons.I0 += amp*brian2.pA
        brian2.run(dur * brian2.ms)
        # stop
        neurons.I0 -= amp*brian2.pA

        # update I trace
        I = np.concatenate([I, amp*np.ones(len(trace.t)-ilast)])
        ilast = len(trace.t)

    # add the delay to finish
    brian2.run(args['delay'] * brian2.ms)

    I = np.concatenate([I, np.zeros(len(trace.t)-ilast)])
    
    # record quantities:
    t = trace.t / brian2.ms
    # I = np.array([args['amp'] if ((tt>args['delay']) & (tt<args['delay']+args['duration'])) else 0 for tt in t])
    
    return trace.t / brian2.ms, trace[0].V[:] / brian2.mV, I, spikes.t / brian2.ms


if __name__=='__main__':

    print(__doc__)
    # starting from an example

    import argparse
    parser=argparse.ArgumentParser(description=
     """ 
     By default the scripts runs the single neuron response to a current step
     """
    ,formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-n', "--NRN", help="NEURON TYPE", default='LIF')
    parser.add_argument('-as', "--amplitudes",
                        help="ARRAY of Amplitude of different steps in pA",\
                        type=float, default=[200, 0, -100], nargs='*')
    parser.add_argument('-ds', "--durations",
                        help="ARRAY of durations of different steps in ms",\
                        type=float, default=[400, 400, 400], nargs='*')
    parser.add_argument('-dl', "--delay",help="Delay before stim onset and for stim ending",
                        type=float, default=150.)
    parser.add_argument("-v", "--verbose", help="",
                        action="store_true")
    args = parser.parse_args()

    # run: 
    t, Vm, I, spikes = current_pulse_sim(vars(args))

    # plot: 
    fig, AX = pt.figure(axes_extents=[[[1,2]],[[1,1]]], figsize=(2,0.8), left=0.5)
    pt.plot(t, Vm, ax=AX[0])
    pt.plot(t, I, ax=AX[1])
    pt.show()
