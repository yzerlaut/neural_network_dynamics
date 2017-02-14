"""
This script connects the different synapses to a target neuron
"""
import brian2
import numpy as np
import itertools, string
import sys
sys.path.append('../')
from cells.cell_library import get_neuron_params
from cells.cell_construct import get_membrane_equation


def build_up_recurrent_connections(Pops, M, SEED=1):
    """
    Construct the synapses from the connectivity matrix
    """
    CONN = np.empty((len(Pops), len(Pops)), dtype=object)
    CONN2 = []

    brian2.seed(SEED)

    for ii, jj in itertools.product(range(len(Pops)), range(len(Pops))):
        if (M[ii,jj]['pconn']>0) and (M[ii,jj]['Q']!=0):
            CONN[ii,jj] = brian2.Synapses(Pops[ii], Pops[jj], model='w:siemens',\
                               on_pre='G'+M[ii,jj]['name']+'_post+=w')
            CONN[ii,jj].connect('i!=j', p=M[ii,jj]['pconn'])
            CONN[ii,jj].w = M[ii,jj]['Q']*brian2.nS
            CONN2.append(CONN[ii,jj])
            
    return CONN2

def build_populations(NTWK, M, with_raster=False, with_pop_act=False, verbose=True):
    """
    sets up the neuronal populations
    """
    POPS = []
    for ntwk, ii in zip(NTWK, range(len(NTWK))):
        if 'params' in ntwk.keys():
            neuron_params = ntwk['params']
        else:
            neuron_params = get_neuron_params(ntwk['type'], number=ntwk['N'], verbose=verbose)
            ntwk['params'] = neuron_params
        POPS.append(get_membrane_equation(neuron_params, M[:,ii]))
        
    if with_pop_act:
        POP_ACT = []
        for pop in POPS:
            POP_ACT.append(brian2.PopulationRateMonitor(pop))
    if with_raster:
        RASTER = []
        for pop in POPS:
            RASTER.append(brian2.SpikeMonitor(pop))

    if with_pop_act and with_raster:
        return POPS, RASTER, POP_ACT
    elif with_raster:
        return POPS, RASTER
    elif with_pop_act:
        return POPS, POP_ACT
    else:
        return POPS

def initialize_to_rest(POPS, NTWK):
    """
    REST BY DEFAULT !

    membrane potential is an absolute value !
    while conductances are relative to leak conductance of the neuron !
    /!\ one population has the same conditions on all its targets !! /!\
    """
    for ii, l in zip(range(len(POPS)), string.ascii_uppercase[:len(POPS)]):
        POPS[ii].V = NTWK[ii]['params']['El']*brian2.mV
        for t in string.ascii_uppercase[:len(POPS)]:
            exec("POPS[ii].G"+t+l+" = 0.*brian2.nS")

def initialize_to_random(POPS, NTWK, G_MATRIX):
    """
    REST BY DEFAULT !

    membrane potential is an absolute value !
    while conductances are relative to leak conductance of the neuron !
    /!\ one population has the same conditions on all its targets !! /!\
    """
    for ii, l in zip(range(len(POPS)), string.ascii_uppercase[:len(POPS)]):
        POPS[ii].V = NTWK[ii]['params']['El']*brian2.mV
        for t in string.ascii_uppercase[:len(POPS)]:
            exec("POPS[ii].G"+t+l+" = 0.*brian2.nS")
            
if __name__=='__main__':

    print(__doc__)
    
    from syn_and_connec_library import get_connectivity_and_synapses_matrix
    import sys, os
    sys.path.append(os.path.expanduser('~')+os.path.sep+'work')
    from graphs.ntwk_dyn_plot import RASTER_PLOT
    
    # starting from an example
    NTWK = [\
            {'name':'exc', 'N':4000, 'type':'LIF'},
            {'name':'inh1', 'N':500, 'type':'LIF'},
            {'name':'inh2', 'N':500, 'type':'LIF'}
    ]
    
    M = get_connectivity_and_synapses_matrix('Vogels-Abbott', number=len(NTWK))
    POPS, RASTER = build_populations(NTWK, M, with_raster=True)

    initialize_to_rest(POPS, NTWK)
    # custom initialization
    for P, t in zip(POPS, ['A', 'B', 'C']):
        P.V = '-60*mV + randn()*5*mV'
        exec("P.GA"+t+" = '(randn() * 1. + 4) * 10.*nS'")
        exec("P.GB"+t+" = '(randn() * 6. + 10.) * 10.*nS'")
        exec("P.GC"+t+" = '(randn() * 6. + 10.) * 10.*nS'")

    SYNAPSES = build_up_recurrent_connections(POPS, M)
    
    net = brian2.Network(brian2.collect())
    net.add(POPS, SYNAPSES, RASTER) # manually add the generated quantities

    net.run(20.*brian2.ms)

    RASTER_PLOT([pop.t/brian2.ms for pop in RASTER], [pop.i for pop in RASTER])

    brian2.show()

    

    
