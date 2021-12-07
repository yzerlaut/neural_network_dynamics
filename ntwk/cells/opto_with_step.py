"""
This file construct the equations for brian2
"""
import numpy as np
import brian2
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from cells.cell_library import get_neuron_params
from cells.cell_construct import get_membrane_equation

def current_pulse_sim_with_opto(args, params=None):
    
    if params is None:
        params = get_neuron_params(args['NRN'])
    params['Vclamp'] = -80
        
    neurons, eqs = get_membrane_equation(params, [],\
                                         return_equations=True)
    if args['verbose']:
        print(eqs)

    fig, ax = brian2.subplots(figsize=(5,3))

    # V value initialization
    neurons.V = params['El']*brian2.mV
    trace = brian2.StateMonitor(neurons, 'V', record=0)
    spikes = brian2.SpikeMonitor(neurons)
    # rest run
    brian2.run(args['delay'] * brian2.ms)
    # first pulse
    neurons.I0 = args['amp']*brian2.pA
    brian2.run(args['duration']/3. * brian2.ms)
    neurons.Gclamp = 1e3*brian2.nS
    brian2.run(args['duration']/3. * brian2.ms)
    neurons.Gclamp = 0*brian2.nS
    brian2.run(args['duration']/3. * brian2.ms)
    # second pulse
    neurons.I0 = 0
    brian2.run(args['delay'] * brian2.ms)
    # We draw nicer spikes
    Vm = trace[0].V[:]
    for t in spikes.t:
        ax.plot(t/brian2.ms*np.ones(2),
                [Vm[int(t/brian2.defaultclock.dt)]/brian2.mV,-10],
                '--', color=args['color'])
    ax.plot(trace.t / brian2.ms, Vm / brian2.mV, color=args['color'])
    
    if 'NRN' in args.keys():
        ax.set_title(args['NRN'])

    ax.annotate(str(int(params['El']))+'mV', (-50,params['El']-5))
    ax.plot([-20], [params['El']], 'k>')
    ax.plot([0,50], [-50, -50], 'k-', lw=4)
    ax.plot([0,0], [-50, -40], 'k-', lw=4)
    ax.annotate('10mV', (-50,-38))
    ax.annotate('50ms', (0,-55))
    # set_plot(ax, [], xticks=[], yticks=[])
    # show()
    if 'save' in args.keys():
        fig.savefig(args['save'])
    return fig
        
if __name__=='__main__':

    print(__doc__)
    import sys
    sys.path.append('../..')
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
    parser.add_argument("--delay",help="Duration of the current step in ms",\
                        type=float, default=100.)
    parser.add_argument('-p', "--post",help="After-Pulse duration of the step (ms)",\
                        type=float, default=400.)
    parser.add_argument("-c", "--color", help="color of the plot",
                        default='k')
    parser.add_argument("--save", help="save the figures with a given string")
    parser.add_argument("-v", "--verbose", help="",
                        action="store_true")
    args = parser.parse_args()

    from graphs.my_graph import set_plot, show
    
    current_pulse_sim_with_opto(vars(args))
    show()

