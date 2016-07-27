"""
This script connects the different synapses to a target neuron
"""
import brian2
import numpy as np
import itertools, string

def build_up_recurrent_connections(Pops, M, SEED=1):
    """
    Construct the synapses from the connectivity matrix
    """
    CONN = np.empty((len(Pops), len(Pops)), dtype=object)

    brian2.seed(SEED)
    # file = open('connectivity_code.py', 'w')
    # file.write('seed('+str(SEED)+') \n')
    for ii, jj in itertools.product(range(len(Pops)), range(len(Pops))):
        CONN[ii,jj] = brian2.Synapses(Pops[ii], Pops[jj], model='w:siemens',\
                               on_pre='G'+M[ii,jj]['name']+'_post+=w')
        CONN[ii,jj].connect('i!=j', p=M[ii,jj]['pconn'])
        CONN[ii,jj].w = M[ii,jj]['Q']*brian2.nS
    return CONN

def build_populations(NTWK, M, with_raster=False):
    """
    sets up the neuronal populations
    """
    POPS = []
    for ntwk, i in zip(NTWK, range(len(NTWK))):
        POPS.append(\
                    get_membrane_equation(\
                    get_neuron_params(ntwk['type'], number=ntwk['N']),
                                          M[:,i]))
    if with_raster:
        RASTER = []
        for pop in POPS:
            RASTER.append(brian2.SpikeMonitor(pop))
        return POPS, RASTER
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
        POPS[ii].V = get_neuron_params(NTWK[ii]['type'])['El']*brian2.mV
        for t in string.ascii_uppercase[:len(POPS)]:
            exec("POPS[ii].G"+t+l+" = 0.*nS")

        
if __name__=='__main__':

    print(__doc__)
    
    # starting from an example

    import sys
    sys.path.append('../')
    from cells.cell_library import get_neuron_params
    from cells.cell_construct import get_membrane_equation
    from syn_and_connec_library import get_connectivity_and_synapses_matrix
    
    NTWK = [\
            {'name':'exc', 'N':4000, 'type':'LIF'},
            {'name':'inh1', 'N':500, 'type':'LIF'},
            {'name':'inh2', 'N':500, 'type':'LIF'}
    ]
    
    M = get_connectivity_and_synapses_matrix('Vogels-Abbott', number=len(NTWK))
    POPS, RASTER = build_populations(NTWK, M, with_raster=True)

    # initialize_to_rest(POPS, NTWK)
    # custom initialization
    for P, t in zip(POPS, ['A', 'B', 'C']):
        P.V = '-60*mV + randn()*5*mV'
        exec("P.GA"+t+" = '(randn() * 1. + 4) * 10.*nS'")
        exec("P.GB"+t+" = '(randn() * 6. + 10.) * 10.*nS'")
        exec("P.GC"+t+" = '(randn() * 6. + 10.) * 10.*nS'")
        
    SYNAPSES = build_up_recurrent_connections(POPS, M)
    
    net = brian2.Network(brian2.collect())
    net.add(POPS, SYNAPSES, RASTER) # manually add the generated quantities

    net.run(50*brian2.ms)

    # plotting 
    fig1 = brian2.figure(figsize=(5,3.5))
    brian2.plot(RASTER[0].t/brian2.ms, RASTER[0].i, '.g',\
         RASTER[1].t/brian2.ms, RASTER[1].i+NTWK[0]['N'], '.r',\
         RASTER[2].t/brian2.ms, RASTER[2].i+NTWK[0]['N']+NTWK[1]['N'], '.r')
    brian2.xlabel('Time (ms)');brian2.ylabel('Neuron index')
    
    fig1.savefig('fig.png')

    

    
