"""
This script connects the different synapses to a target neuron
"""
import brian2
import numpy as np
import itertools, string, sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from cells.cell_library import get_neuron_params
from cells.cell_construct import get_membrane_equation

def collect_and_run(NTWK, tstop=100, dt=0.1):
    NTWK['dt'], NTWK['tstop'] = dt, tstop
    brian2.defaultclock.dt = dt*brian2.ms
    net = brian2.Network(brian2.collect())
    OBJECT_LIST = []
    for key in ['POPS', 'REC_SYNAPSES', 'RASTER',
                'POP_ACT', 'VMS', 'ISYNe', 'ISYNi',
                'PRE_SPIKES', 'PRE_SYNAPSES']:
        if key in NTWK.keys():
            net.add(NTWK[key])
    net.run(tstop*brian2.ms)
    return net

def build_up_recurrent_connections(NTWK, SEED=1):
    """
    Construct the synapses from the connectivity matrix
    """
    CONN = np.empty((len(NTWK['POPS']), len(NTWK['POPS'])), dtype=object)
    CONN2 = []

    brian2.seed(SEED)

    for ii, jj in itertools.product(range(len(NTWK['POPS'])), range(len(NTWK['POPS']))):
        if (NTWK['M'][ii,jj]['pconn']>0) and (NTWK['M'][ii,jj]['Q']!=0):
            CONN[ii,jj] = brian2.Synapses(NTWK['POPS'][ii], NTWK['POPS'][jj], model='w:siemens',\
                               on_pre='G'+NTWK['M'][ii,jj]['name']+'_post+=w')
            CONN[ii,jj].connect('i!=j', p=NTWK['M'][ii,jj]['pconn'])
            CONN[ii,jj].w = NTWK['M'][ii,jj]['Q']*brian2.nS
            CONN2.append(CONN[ii,jj])

    NTWK['REC_SYNAPSES'] = CONN2

def build_populations(NEURONS, M, with_raster=False, with_pop_act=False, with_Vm=0,
                      verbose=False, with_synaptic_currents=False, with_synaptic_conductances=False):
    """
    sets up the neuronal populations
    and  construct a network object containing everything
    """
    
    NTWK = {'NEURONS':NEURONS, 'M':M}
    
    NTWK['POPS'] = []
    for nrn, ii in zip(NEURONS, range(len(NTWK))):
        if 'params' in nrn.keys():
            # to have a population with custom params !
            neuron_params = nrn['params']
        else:
            neuron_params = get_neuron_params(nrn['type'], number=nrn['N'],
                                              name=nrn['name'],
                                              verbose=verbose)
            nrn['params'] = neuron_params
        NTWK['POPS'].append(get_membrane_equation(neuron_params, M[:,ii],
                                          with_synaptic_currents=with_synaptic_currents,
                                          with_synaptic_conductances=with_synaptic_conductances,
                                          verbose=verbose))
    if with_pop_act:
        NTWK['POP_ACT'] = []
        for pop in NTWK['POPS']:
            NTWK['POP_ACT'].append(brian2.PopulationRateMonitor(pop))
    if with_raster:
        NTWK['RASTER'] = []
        for pop in NTWK['POPS']:
            NTWK['RASTER'].append(brian2.SpikeMonitor(pop))
    if with_Vm>0:
        NTWK['VMS'] = []
        for pop in NTWK['POPS']:
            NTWK['VMS'].append(brian2.StateMonitor(pop, 'V', record=np.arange(with_Vm)))
    if with_synaptic_currents:
        NTWK['ISYNe'], NTWK['ISYNi'] = [], []
        for pop in NTWK['POPS']:
            NTWK['ISYNe'].append(brian2.StateMonitor(pop, 'Ie', record=np.arange(max([1,with_Vm]))))
            NTWK['ISYNi'].append(brian2.StateMonitor(pop, 'Ii', record=np.arange(max([1,with_Vm]))))
    if with_synaptic_conductances:
        NTWK['Ge'], NTWK['Gi'] = [], []
        for pop in NTWK['POPS']:
            NTWK['Ge'].append(brian2.StateMonitor(pop, 'Ge', record=np.arange(max([1,with_Vm]))))
            NTWK['Gi'].append(brian2.StateMonitor(pop, 'Gi', record=np.arange(max([1,with_Vm]))))

    NTWK['PRE_SPIKES'], NTWK['PRE_SYNAPSES'] = [], [] # in case of afferent inputs
    return NTWK

def initialize_to_rest(NTWK):
    """
    Vm to resting potential and conductances to 0
    """
    for ii in range(len(NTWK['POPS'])):
        NTWK['POPS'][ii].V = NTWK['NEURONS'][ii]['params']['El']*brian2.mV
        for jj in range(len(NTWK['POPS'])):
            if NTWK['M'][jj,ii]['pconn']>0: # if connection
                exec("NTWK['POPS'][ii].G"+NTWK['M'][jj,ii]['name']+" = 0.*brian2.nS")

                
def initialize_to_random(NTWK, Gmean=10., Gstd=3.):
    """

    membrane potential is an absolute value !
    while conductances are relative to leak conductance of the neuron !
    /!\ one population has the same conditions on all its targets !! /!\
    """
    for ii in range(len(NTWK['POPS'])):
        NTWK['POPS'][ii].V = NTWK[ii]['params']['El']*brian2.mV
        for jj in range(len(NTWK['POPS'])):
            if NTWK['M'][jj,ii]['pconn']>0: # if connection
                exec("NTWK['POPS'][ii].G"+NTWK['M'][jj,ii]['name']+\
                     " = ("+str(Gmean)+"+brian2.randn()*"+str(Gstd)+")*brian2.nS")
            
