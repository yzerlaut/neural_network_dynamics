"""
This file construct the equations for brian2
"""
import numpy as np
import brian2
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from neural_network_dynamics.cells.cell_library import get_neuron_params, built_up_neuron_params

def get_membrane_equation(neuron_params, synaptic_array,\
                          return_equations=False,
                          with_synaptic_currents=False,
                          with_synaptic_conductances=False,
                          verbose=False):

    if verbose:
        print('------------------------------------------------------------------')
        print('==> Neuron(s) with parameters')
        print(neuron_params)
        print('------------------------------------------------------------------')
        
    ## pure membrane equation
    if ('deltaV' not in neuron_params) or (neuron_params['deltaV']==0):
        # if hard threshold : Integrate and Fire
        eqs = """
        dV/dt = (%(Gl)f*nS*(%(El)f*mV - V) + I - w_adapt)/(%(Cm)f*pF) : volt (unless refractory) """ % neuron_params
    else:
        eqs = """
        dV/dt = (%(Gl)f*nS*(%(El)f*mV - V) + %(Gl)f*nS*%(deltaV)f*mV*exp(-(%(Vthre)f*mV-V)/(%(deltaV)f*mV)) + I - w_adapt)/(%(Cm)f*pF) : volt (unless refractory) """ % neuron_params

    ## Adaptation current
    if (neuron_params['a']!=0) and (neuron_params['b']!=0): # adaptation current or not ?
        eqs += """
        dw_adapt/dt = ( -%(a)f*nS*( %(El)f*mV - V) - w_adapt )/(%(tauw)f*ms) : amp  """ % neuron_params
    else:
        eqs += """
        w_adapt : amp  """

    ## --> starting current definition
    eqs += """
        I = I0 """
    
    ## intrinsic currents
    if 'Ioscill_amp' in neuron_params:
        eqs += '+ %(Ioscill_amp)f*pA *(1 - cos(2 * pi * %(Ioscill_freq)f * Hz * t))/2 ' % neuron_params
        
    ## synaptic currents, 1) adding all synaptic currents to the membrane equation via the I variable
    for synapse in synaptic_array:
        if synapse['pconn']>0:
            # loop over each presynaptic element onto this target
            Gsyn = 'G'+synapse['name']
            if 'alpha' in synapse:
                eqs += '+'+Gsyn+'*( %(alpha)f*(%(Erev)f*mV - V) + (1.0-%(alpha)f)*(%(Erev)f*mV - %(V0)f*mV) )' % synapse
                # print('using conductance-current mixture in synaptic equations, with ratio', synapse['alpha'])
            else:
                eqs += '+'+Gsyn+'*(%(Erev)f*mV - V)' % synapse
    # adding a potential clamping current
    if 'Vclamp' in neuron_params:
        eqs += ' + Gclamp * (%(Vclamp)f*mV - V)' % neuron_params
    eqs += ' : amp'
    ## ending current definition <--
    if 'Vclamp' in neuron_params:
        eqs += """
        Gclamp : siemens """

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
        
    # adexp, pratical detection threshold Vthre+5*deltaV
    neurons = brian2.NeuronGroup(neuron_params['N'], model=eqs,
               method='euler', refractory=str(neuron_params['Trefrac'])+'*ms',
               threshold='V>'+str(neuron_params['Vthre']+5.*neuron_params['deltaV'])+'*mV',
               reset='V='+str(neuron_params['Vreset'])+'*mV; w_adapt+='+str(neuron_params['b'])+'*pA')

    if return_equations:
        return neurons, eqs
    else:
        return neurons

        
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

    from neural_network_dynamics.cells.pulse_protocols import current_pulse_sim
    from graphs.my_graph import graphs
    mg = graphs()
    
    mg.response_to_current_pulse(*current_pulse_sim(vars(args)))
    mg.show()

