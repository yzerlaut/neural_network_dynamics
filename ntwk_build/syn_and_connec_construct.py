"""
This script connects the different synapses to a target neuron
"""
import brian2
import numpy as np
import itertools, string
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from cells.cell_library import get_neuron_params
from cells.cell_construct import get_membrane_equation

def collect_and_run(OBJECT_LIST, tstop=100, dt=0.1):
    brian2.defaultclock.dt = dt*brian2.ms
    net = brian2.Network(brian2.collect())
    net.add(*OBJECT_LIST)
    net.run(tstop*brian2.ms)
    return net

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

def build_populations(NTWK, M, with_raster=False, with_pop_act=False, with_Vm=0,
                      verbose=False, with_synaptic_currents=False):
    """
    sets up the neuronal populations
    """
    POPS = []
    for ntwk, ii in zip(NTWK, range(len(NTWK))):
        if 'params' in ntwk.keys():
            # to have a population with custom params !
            neuron_params = ntwk['params']
        else:
            neuron_params = get_neuron_params(ntwk['type'], number=ntwk['N'],
                                              name=ntwk['name'],
                                              verbose=verbose)
            ntwk['params'] = neuron_params
        POPS.append(get_membrane_equation(neuron_params, M[:,ii],
                                          with_synaptic_currents=with_synaptic_currents,
                                          verbose=verbose))
    if with_pop_act:
        POP_ACT = []
        for pop in POPS:
            POP_ACT.append(brian2.PopulationRateMonitor(pop))
    if with_raster:
        RASTER = []
        for pop in POPS:
            RASTER.append(brian2.SpikeMonitor(pop))
    if with_Vm>0:
        VMS = []
        for pop in POPS:
            VMS.append(brian2.StateMonitor(pop, 'V', record=np.arange(with_Vm)))
    if with_synaptic_currents:
        ISYNe, ISYNi = [], []
        for pop in POPS:
            ISYNe.append(brian2.StateMonitor(pop, 'Ie', record=np.arange(max([1,with_Vm]))))
            ISYNi.append(brian2.StateMonitor(pop, 'Ii', record=np.arange(max([1,with_Vm]))))

    if with_pop_act and with_raster and with_Vm>0 and with_synaptic_currents:
        return POPS, RASTER, POP_ACT, VMS, ISYNe, ISYNi
    if with_pop_act and with_Vm>0 and with_synaptic_currents:
        return POPS, POP_ACT, VMS, ISYNe, ISYNi
    elif with_raster and with_Vm>0 and with_synaptic_currents:
        return POPS, RASTER, VMS, ISYNe, ISYNi
    elif with_pop_act and with_raster:
        return POPS, RASTER, POP_ACT
    elif with_pop_act and with_Vm>0:
        return POPS, POP_ACT, VMS
    elif with_raster and with_Vm>0:
        return POPS, RASTER, VMS
    elif with_Vm>0 and with_synaptic_currents:
        return POPS, VMS, ISYNe, ISYNi
    elif with_raster:
        return POPS, RASTER
    elif with_pop_act:
        return POPS, POP_ACT
    elif with_Vm>0:
        return POPS, VMS
    else:
        return POPS

def initialize_to_rest(POPS, NTWK, M):
    """
    Vm to resting potential and conductances to 0
    """
    for ii in range(len(POPS)):
        POPS[ii].V = NTWK[ii]['params']['El']*brian2.mV
        for jj in range(len(POPS)):
            if M[jj,ii]['pconn']>0: # if connection
                exec("POPS[ii].G"+M[jj,ii]['name']+" = 0.*brian2.nS")

                
def initialize_to_random(POPS, NTWK, M, Gmean=10., Gstd=3.):
    """

    membrane potential is an absolute value !
    while conductances are relative to leak conductance of the neuron !
    /!\ one population has the same conditions on all its targets !! /!\
    """
    for ii in range(len(POPS)):
        POPS[ii].V = NTWK[ii]['params']['El']*brian2.mV
        for jj in range(len(POPS)):
            if M[jj,ii]['pconn']>0: # if connection
                exec("POPS[ii].G"+M[jj,ii]['name']+\
                     " = ("+str(Gmean)+"+brian2.randn()*"+str(Gstd)+")*brian2.nS")
            
if __name__=='__main__':
    _=0
    
