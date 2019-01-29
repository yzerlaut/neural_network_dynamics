import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from graphs.my_graph import *
from neural_network_dynamics.cells.cell_library import *
from neural_network_dynamics.cells.cell_construct import *

def current_pulse_sim(args, params=None, verbose=False):
    
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
    if ('amplitudes' in args) and len(args['amplitudes'])>0:
        if len(args['durations'])==len(args['amplitudes']):
            durations = args['durations']
        else:
            durations = args['duration']*np.ones(len(args['amplitudes']))
        for amp, dur in zip(args['amplitudes'], durations):
            neurons.I0 += amp*brian2.pA
            brian2.run(dur * brian2.ms)
            neurons.I0 -= amp*brian2.pA
    else:
        # start pulse
        neurons.I0 += args['amp']*brian2.pA
        brian2.run(args['duration'] * brian2.ms)
        # end pulse
        neurons.I0 -= args['amp']*brian2.pA
    brian2.run(args['delay'] * brian2.ms)

    t = trace.t / brian2.ms
    I = np.array([args['amp'] if ((tt>args['delay']) & (tt<args['delay']+args['duration'])) else 0 for tt in t])
    
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
    parser.add_argument('-a', "--amp",help="Amplitude of the current in pA",\
                        type=float, default=200.)
    parser.add_argument('-d', "--duration",help="Duration of the current step in ms",\
                        type=float, default=400.)
    parser.add_argument('-as', "--amplitudes",
                        help="ARRAY of Amplitude of different steps in pA",\
                        type=float, default=[], nargs='*')
    parser.add_argument('-ds', "--durations",
                        help="ARRAY of durations of different steps in ms",\
                        type=float, default=[], nargs='*')
    parser.add_argument('-dl', "--delay",help="Duration of the current step in ms",\
                        type=float, default=150.)
    parser.add_argument('-p', "--post",help="After-Pulse duration of the step (ms)",\
                        type=float, default=400.)
    parser.add_argument("-c", "--color", help="color of the plot",
                        default='k')
    parser.add_argument("--save", default='', help="save the figures with a given string")
    parser.add_argument("-v", "--verbose", help="",
                        action="store_true")
    args = parser.parse_args()

    from graphs.my_graph import set_plot, show
    from graphs.single_cell_plots import *
    # response_to_current_pulse(*current_pulse_sim(vars(args)))
    # VMS, II, SPIKES = [], [], []
    # for amp in [-50, 50, 200]:
    #     args.amp = amp
    #     t, Vm, I, spikes = current_pulse_sim(vars(args))
    #     VMS.append(Vm)
    #     II.append(I)
    #     SPIKES.append(spikes)
    # response_to_multiple_current_pulse(t, VMS, II, SPIKES)
    # show()
    
    for delta in [0., 1., 2., 4.]:
        args.NRN = 'EIF_deltaV_'+str(delta)
        VMS, II, SPIKES = [], [], []
        for amp in [-150, 50, 250]:
            args.amp = amp
            t, Vm, I, spikes = current_pulse_sim(vars(args))
            VMS.append(Vm)
            II.append(I)
            SPIKES.append(spikes)
        fig, ax = response_to_multiple_current_pulse(t, VMS, II, SPIKES)
        fig.suptitle('$\delta$=%imV' % delta)
        fig.savefig(desktop+'fig+%i.svg' % delta)
