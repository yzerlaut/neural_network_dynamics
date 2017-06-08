"""
This file construct the equations for brian2
"""
import numpy as np
import brian2

def get_membrane_equation(neuron_params, synaptic_array,\
                          return_equations=False, with_synaptic_currents=False,
                          with_synaptic_conductances=False,
                          verbose=False):

    if verbose:
        print('------------------------------------------------------------------')
        print('==> Neuron(s) with parameters')
        print(neuron_params)
        print('------------------------------------------------------------------')
    ## pure membrane equation
    if neuron_params['delta_v']==0:
        # if hard threshold : Integrate and Fire
        eqs = """
        dV/dt = (%(Gl)f*nS*(%(El)f*mV - V) + I - w_adapt)/(%(Cm)f*pF) : volt (unless refractory) """ % neuron_params
    else:
        eqs = """
        dV/dt = (%(Gl)f*nS*(%(El)f*mV - V) + %(Gl)f*nS*%(delta_v)f*mV*exp(-(%(Vthre)f*mV-V)/(%(delta_v)f*mV)) + I - w_adapt)/(%(Cm)f*pF) : volt (unless refractory) """ % neuron_params

    ## Adaptation current
    if (neuron_params['a']!=0) and (neuron_params['b']!=0): # adaptation current or not ?
        eqs += """
        dw_adapt/dt = ( -%(a)f*nS*( %(El)f*mV - V) - w_adapt )/(%(tauw)f*ms) : amp  """ % neuron_params
    else:
        eqs += """
        w_adapt : amp  """

    ## synaptic currents, 1) adding all synaptic currents to the membrane equation via the I variable
    eqs += """
        I = I0 """
    for synapse in synaptic_array:
        if synapse['pconn']>0:
            # loop over each presynaptic element onto this target
            Gsyn = 'G'+synapse['name']
            eqs += '+'+Gsyn+'*(%(Erev)f*mV - V)' % synapse
    eqs += ' : amp'

    ## synaptic currents, 2) constructing the temporal dynamics of the synaptic conductances
    ## N.B. VALID ONLY FOR EXPONENTIAL SYNAPSES UNTIL NOW !!!!
    for synapse in synaptic_array:
        # loop over each presynaptic element onto this target
        if synapse['pconn']>0:
            Gsyn = 'G'+synapse['name']
            eqs += """
            """+'d'+Gsyn+'/dt = -'+Gsyn+'*(1./(%(Tsyn)f*ms)) : siemens' % synapse
    eqs += """
        I0 : amp """

    if with_synaptic_currents:
        # compute excitatory currents
        eqs += """
        Ie = 0*pA """
        for synapse in synaptic_array:
            if (synapse['Erev']>-20) and (synapse['pconn']>0): # if excitatory
                # loop over each presynaptic element onto this target
                Gsyn = 'G'+synapse['name']
                eqs += '+'+Gsyn+'*(%(Erev)f*mV - V)' % synapse
        eqs += ' : amp' # no synaptic currents when clamped at the spiking level
        # compute inhibitory currents
        eqs += """
        Ii = 0*pA """
        for synapse in synaptic_array:
            if (synapse['Erev']<-60) and (synapse['pconn']>0): # if inhibitory
                # loop over each presynaptic element onto this target
                Gsyn = 'G'+synapse['name']
                eqs += '+'+Gsyn+'*(%(Erev)f*mV - V)' % synapse
        eqs += ' : amp' # no synaptic currents when clamped at the spiking level

    if with_synaptic_conductances:
        # compute excitatory conductances
        eqs += """
        Ge = 0*nS """ # need an expression of V for update in brian2
        for synapse in synaptic_array:
            if (synapse['Erev']>-20) and (synapse['pconn']>0): # if excitatory
                eqs += '+G'+synapse['name']
        eqs += ' : siemens' 
        # compute inhibitory conductances
        eqs += """
        Gi = 0*nS """
        for synapse in synaptic_array:
            if (synapse['Erev']<-60) and (synapse['pconn']>0): # if inhibitory
                eqs += '+G'+synapse['name']
        eqs += ' : siemens' 

    if verbose:
        print(eqs)
        
    # adexp, pratical detection threshold Vthre+5*delta_v
    neurons = brian2.NeuronGroup(neuron_params['N'], model=eqs,
               method='euler', refractory=str(neuron_params['Trefrac'])+'*ms',
               threshold='V>'+str(neuron_params['Vthre']+5.*neuron_params['delta_v'])+'*mV',
               reset='V='+str(neuron_params['Vreset'])+'*mV; w_adapt+='+str(neuron_params['b'])+'*pA')

    if return_equations:
        return neurons, eqs
    else:
        return neurons

def current_pulse_sim(args, params=None):
    
    import brian2
    from cells.cell_library import get_neuron_params
    from graphs.my_graph import set_plot

    if params is None:
        params = get_neuron_params(args['NRN'])
        
    neurons, eqs = get_membrane_equation(params, [],\
                                         return_equations=True)

    fig, ax = brian2.subplots(figsize=(5,3))

    # V value initialization
    neurons.V = params['El']*brian2.mV
    trace = brian2.StateMonitor(neurons, 'V', record=0)
    spikes = brian2.SpikeMonitor(neurons)
    brian2.run(100 * brian2.ms)
    neurons.I0 = args['amp']*brian2.pA
    brian2.run(args['duration'] * brian2.ms)
    neurons.I0 = 0*brian2.pA
    brian2.run(200 * brian2.ms)
    # We draw nicer spikes
    V = trace[0].V[:]
    for t in spikes.t:
        ax.plot(t/brian2.ms*np.ones(2), [V[int(t/brian2.defaultclock.dt)]/brian2.mV+2,-10], '--',\
                 color=args['color'])
    ax.plot(trace.t / brian2.ms, V / brian2.mV, color=args['color'])

    if 'NRN' in args.keys():
        ax.set_title(args['NRN'])

    ax.annotate(str(int(params['El']))+'mV', (-50,params['El']-5))
    ax.plot([-20], [params['El']], 'k>')
    ax.plot([0,50], [-50, -50], 'k-', lw=4)
    ax.plot([0,0], [-50, -40], 'k-', lw=4)
    ax.annotate('10mV', (-50,-38))
    ax.annotate('50ms', (0,-55))
    set_plot(ax, [], xticks=[], yticks=[])
    if 'save' in args.keys():
        fig.savefig(['save'])
    return fig
        
        
if __name__=='__main__':

    print(__doc__)
    
    # starting from an example

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
    parser.add_argument("--save", help="save the figures with a given string")
    args = parser.parse_args()


    current_pulse_sim(vars(args))
    plt.show()

