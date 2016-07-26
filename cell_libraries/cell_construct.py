"""
This file construct the equations for brian2
"""
import numpy as np
import brian2

def get_membrane_equation(neuron_params, synaptic_array,\
                          return_equations=False):

    ## pure membrane equation
    if neuron_params['delta_v']==0:
        # if hard threshold : Integrate and Fire
        eqs = """
        dV/dt = (%(Gl)f*nS*(%(El)f*mV - V) + I - w_adapt)/(%(Cm)f*pF) : volt (unless refractory) """ % neuron_params
    else:
        eqs = """
        dV/dt = (%(Gl)f*nS*(%(El)f*mV - V) + %(Gl)f*nS*%(delta_v)f*mV*exp(-(%(Vthre)f*mV-V)/(%(delta_v)f*mV)) + I - w_adapt)/(%(Cm)f*pF) : volt (unless refractory) """ % neuron_params

    ## Adaptation current
    if neuron_params['tauw']>0: # adaptation current or not ?
        eqs += """
        dw_adapt/dt = ( -%(a)f*nS*( %(El)f*mV - V) - w_adapt )/(%(tauw)f*ms) : amp  """ % neuron_params
    else:
        eqs += """
        w_adapt : amp  """

    ## synaptic currents, 1) adding all synaptic currents to the membrane equation via the I variable
    eqs += """
        I = I0 """
    for synapse in synaptic_array:
        # loop over each presynaptic element onto this target
        Gsyn = 'G'+synapse['name']
        eqs += '+'+Gsyn+'*(%(Erev)f*mV - V)' % synapse
    eqs += ' : amp'
    
    ## synaptic currents, 2) constructing the temporal dynamics of the synaptic conductances
    ## N.B. VALID ONLY FOR EXPONENTIAL SYNAPSES UNTIL NOW !!!!
    for synapse in synaptic_array:
        # loop over each presynaptic element onto this target
        Gsyn = 'G'+synapse['name']
        eqs += """
        """+'d'+Gsyn+'/dt = -'+Gsyn+'*(1./(%(Tsyn)f*ms)) : siemens' % synapse
    eqs += """
        I0 : amp """
    # adexp, pratical detection threshold Vthre+5*delta_v
    neurons = brian2.NeuronGroup(neuron_params['N'], model=eqs,\
                                 refractory=str(neuron_params['Trefrac'])+'*ms',
                                 threshold='V>'+str(neuron_params['Vthre']+5.*neuron_params['delta_v'])+'*mV',
                                 reset='V='+str(neuron_params['Vreset'])+'*mV; w_adapt+='+str(neuron_params['b'])+'*pA')
                                 

    if return_equations:
        return neurons, eqs
    else:
        return neurons

        
if __name__=='__main__':

    print(__doc__)
    
    # starting from an example

    from brian2 import *
    from cell_library import get_neuron_params
    import sys
    sys.path.append('../')
    from common_libraries.graphs.my_graph import set_plot

    import argparse
    parser=argparse.ArgumentParser(description=
     """ 
     Generating random sample of a given distributions and
     comparing it with its theoretical value
     """
    ,formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-n', "--NRN", help="NEURON TYPE", default='LIF')
    parser.add_argument('-a', "--amp",help="Amplitude of the current in pA",\
                        type=float, default=200.)
    parser.add_argument('-d', "--duration",help="Duration of the current step in ms",\
                        type=float, default=400.)
    parser.add_argument('-p', "--post",help="After-Pulse duration of the step (ms)",\
                        type=float, default=400.)
    parser.add_argument("-c", "--color", help="color of the plot",
                        default='k')
    parser.add_argument("-s", "--save", help="save the figures",
                        action="store_true")
    args = parser.parse_args()
    
    neurons, eqs = get_membrane_equation(get_neuron_params(args.NRN), [],\
                                       return_equations=True)

    fig, ax = plt.subplots(figsize=(5,3))
    
    # V value initialization
    neurons.V = -65.*mV
    trace = StateMonitor(neurons, 'V', record=0)
    spikes = SpikeMonitor(neurons)
    run(100 * ms)
    neurons.I0 = args.amp*pA
    run(args.duration * ms)
    neurons.I0 = 0*pA
    run(200 * ms)
    # We draw nicer spikes
    V = trace[0].V[:]
    for t in spikes.t:
        plt.plot(t/ms*np.ones(2), [V[int(t/defaultclock.dt)]/mV+2,-10], '--',\
                 color=args.color)
    ax.plot(trace.t / ms, V / mV, color=args.color)

    ax.set_title(args.NRN)

    ax.annotate('-65mV', (-50,-70))
    ax.plot([-20], [-65], 'k>')
    ax.plot([0,50], [-50, -50], 'k-', lw=4)
    ax.plot([0,0], [-50, -40], 'k-', lw=4)
    ax.annotate('10mV', (-50,-38))
    ax.annotate('50ms', (0,-55))
    set_plot(ax, [], xticks=[], yticks=[])
    fig.savefig('fig.png', dpi=100)
    

    
