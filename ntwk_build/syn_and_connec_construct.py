"""
This script connects the different synapses to a target neuron
"""
import brian2
import numpy as np
import itertools, string, sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from cells.cell_library import get_neuron_params
from cells.cell_construct import get_membrane_equation
from cells.cell_construct import built_up_neuron_params

def collect_and_run(NTWK, verbose=False):
    """
    /!\ When you add a new object, THINK ABOUT ADDING IT TO THE COLLECTION !  /!\
    """
    NTWK['dt'], NTWK['tstop'] = NTWK['Model']['dt'], NTWK['Model']['tstop'] 
    brian2.defaultclock.dt = NTWK['dt']*brian2.ms
    net = brian2.Network(brian2.collect())
    OBJECT_LIST = []
    for key in ['POPS', 'REC_SYNAPSES', 'RASTER',
                'POP_ACT', 'VMS', 'ISYNe', 'ISYNi',
                'GSYNe', 'GSYNi',
                'PRE_SPIKES', 'PRE_SYNAPSES']:
        if key in NTWK.keys():
            net.add(NTWK[key])
    if verbose:
        print('running simulation [...]')
    net.run(NTWK['tstop']*brian2.ms)
    if verbose:
        print('-> done !')
    return net

def build_up_recurrent_connections(NTWK, SEED=1, verbose=False):
    """
    Construct the synapses from the connectivity matrix
    """
    CONN = np.empty((len(NTWK['POPS']), len(NTWK['POPS'])), dtype=object)
    CONN2 = []

    # brian2.seed(SEED)
    np.random.seed(SEED)

    if verbose:
        print('drawing random connections [...]')
        
    for ii, jj in itertools.product(range(len(NTWK['POPS'])), range(len(NTWK['POPS']))):
        if (NTWK['M'][ii,jj]['pconn']>0) and (NTWK['M'][ii,jj]['Q']!=0):
            CONN[ii,jj] = brian2.Synapses(NTWK['POPS'][ii], NTWK['POPS'][jj], model='w:siemens',\
                               on_pre='G'+NTWK['M'][ii,jj]['name']+'_post+=w')
            # CONN[ii,jj].connect(p=NTWK['M'][ii,jj]['pconn'], condition='i!=j')
            # N.B. the brian2 settings does weird things (e.g. it creates synchrony)
            # so we draw manually the connection to fix synaptic numbers
            N_per_cell = int(NTWK['M'][ii,jj]['pconn']*NTWK['POPS'][ii].N)
            if ii==jj: # need to take care of no autapse
                i_rdms = np.concatenate([\
                                np.random.choice(
                                    np.delete(np.arange(NTWK['POPS'][ii].N), [iii]), N_per_cell)\
                                          for iii in range(NTWK['POPS'][jj].N)])
            else:
                i_rdms = np.concatenate([\
                                np.random.choice(np.arange(NTWK['POPS'][ii].N), N_per_cell)\
                                          for jjj in range(NTWK['POPS'][jj].N)])
            j_fixed = np.concatenate([np.ones(N_per_cell,dtype=int)*jjj for jjj in range(NTWK['POPS'][jj].N)])
            CONN[ii,jj].connect(i=i_rdms, j=j_fixed) 
            CONN[ii,jj].w = NTWK['M'][ii,jj]['Q']*brian2.nS
            CONN2.append(CONN[ii,jj])

    NTWK['REC_SYNAPSES'] = CONN2


def get_syn_and_conn_matrix(Model, POPULATIONS,
                            AFFERENT_POPULATIONS=[],
                            SI_units=False, verbose=False):

    SOURCE_POPULATIONS = POPULATIONS+AFFERENT_POPULATIONS
    N = len(POPULATIONS)
    Naff = len(SOURCE_POPULATIONS)
    
    # creating empty arry of objects (future dictionnaries)
    M = np.empty((len(SOURCE_POPULATIONS), len(POPULATIONS)), dtype=object)
    # default initialisation
    for i, j in itertools.product(range(len(SOURCE_POPULATIONS)), range(len(POPULATIONS))):
        source_pop, target_pop = SOURCE_POPULATIONS[i], POPULATIONS[j]
        if len(source_pop.split('Exc'))>1:
            Erev, Ts = Model['Ee'], Model['Tse']
        elif len(source_pop.split('Inh'))>1:
            Erev, Ts = Model['Ei'], Model['Tsi']
        else:
            print(' /!\ AFFERENT POP COULD NOT BE CLASSIFIED AS Exc or Inh /!\ ')
            print('-----> set to Exc by default')
            Erev, Ts = Model['Ee'], Model['Tse']

        if ('p_'+source_pop+'_'+target_pop in Model.keys()) and ('Q_'+source_pop+'_'+target_pop in Model.keys()):
            pconn, Qsyn = Model['p_'+source_pop+'_'+target_pop], Model['Q_'+source_pop+'_'+target_pop]
        else:
            if verbose:
                print('No connection for:', source_pop,'->', target_pop)
            pconn, Qsyn = 0., 0.
                
        M[i, j] = {'pconn': pconn, 'Q': Qsyn,
                   'Erev': Erev, 'Tsyn': Ts,
                   'name':source_pop+target_pop}

    if SI_units:
        print('synaptic network parameters in SI units')
        for m in M.flatten():
            m['Q'] *= 1e-9
            m['Erev'] *= 1e-3
            m['Tsyn'] *= 1e-3
    else:
        if verbose:
            print('synaptic network parameters --NOT-- in SI units')

    return M
    
def build_populations(Model, POPULATIONS,
                      AFFERENT_POPULATIONS=[],
                      with_raster=False, with_pop_act=False,
                      with_Vm=0, with_synaptic_currents=False,
                      with_synaptic_conductances=False,
                      NEURONS=None,
                      verbose=False):
    """
    sets up the neuronal populations
    and  construct a network object containing everything
    """

    ## NEURONS AND CONNECTIVITY MATRIX
    if NEURONS is None:
        NEURONS = []
        for pop in POPULATIONS:
            NEURONS.append({'name':pop, 'N':Model['N_'+pop]})

    ########################################################################
    ####    TO BE WRITTEN 
    ########################################################################

    
    NTWK = {'NEURONS':NEURONS, 'Model':Model,
            'POPULATIONS':np.array(POPULATIONS),
            'M':get_syn_and_conn_matrix(Model, POPULATIONS,
                                        AFFERENT_POPULATIONS=AFFERENT_POPULATIONS,
                                        verbose=verbose)}
    
    NTWK['POPS'] = []
    for ii, nrn in enumerate(NEURONS):
        neuron_params = built_up_neuron_params(Model, nrn['name'], N=nrn['N'])
        NTWK['POPS'].append(get_membrane_equation(neuron_params, NTWK['M'][:,ii],
                                                  with_synaptic_currents=with_synaptic_currents,
                                                  with_synaptic_conductances=with_synaptic_conductances,
                                                  verbose=verbose))
        nrn['params'] = neuron_params

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
        NTWK['GSYNe'], NTWK['GSYNi'] = [], []
        for pop in NTWK['POPS']:
            NTWK['GSYNe'].append(brian2.StateMonitor(pop, 'Ge', record=np.arange(max([1,with_Vm]))))
            NTWK['GSYNi'].append(brian2.StateMonitor(pop, 'Gi', record=np.arange(max([1,with_Vm]))))

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
            
