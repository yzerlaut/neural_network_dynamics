"""
This script connects the different synapses to a target neuron
"""
import brian2
import numpy as np
import itertools, string, sys, pathlib

from ..cells.cell_library import get_neuron_params
from ..cells.cell_construct import get_membrane_equation
from ..cells.cell_construct import built_up_neuron_params

def collect_and_run(NTWK, verbose=False, INTERMEDIATE_INSTRUCTIONS=[]):
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

    current_t = 0
    for instrct in INTERMEDIATE_INSTRUCTIONS:
        # we run the simulation until that instruction
        tdur = instrct['time']-current_t
        net.run(tdur*brian2.ms)
        # execute instruction:
        instrct['function'](NTWK)
        current_t = instrct['time']
    net.run((NTWK['tstop']-current_t)*brian2.ms)
    if verbose:
        print('-> done !')
    return net

def build_up_recurrent_connections(NTWK, SEED=1,
                                   with_ring_geometry=False,
                                   store_connections=False,
                                   verbose=False):
    """
    Construct the synapses from the connectivity matrix
    """
    
    # brian2.seed(SEED)
    np.random.seed(SEED)

    if verbose:
        print('drawing random connections [...]')

    if with_ring_geometry:
        NTWK['REC_SYNAPSES'] = random_distance_dependent_connections(NTWK)
    else:
        NTWK['REC_SYNAPSES'] = random_connections(NTWK, store_connections=store_connections)


def build_fixed_aff_to_pop_matrix(afferent_pop, target_pop,
                                  Model,
                                  N_source_pop=None,
                                  N_target_pop=None,
                                  SEED=3):
    """
    Generates the connectivity matrix for a random projection from 
    one population to the other !

    possibility to subsample the target or source pop through the "N_source_pop" and "N_target_pop" args
    """
    np.random.seed(SEED) # insure a precise seed !

    if N_source_pop is None:
        N_source_pop = Model['N_'+afferent_pop]
    if N_target_pop is None:
        N_target_pop = Model['N_'+target_pop]
        
    Nsyn_onto_target_from_source = int(Model['p_'+afferent_pop+'_'+target_pop]*Model['N_'+target_pop])
    
    return np.array([\
      np.random.choice(np.arange(N_target_pop), Nsyn_onto_target_from_source, replace=False)\
                     for k in range(N_source_pop)], dtype=int)

def build_fixed_afference(NTWK,
                          AFFERENT_POPULATIONS,
                          TARGET_POPULATIONS, SEED=1):
    
    for i, afferent_pop in enumerate(AFFERENT_POPULATIONS):
        for j, target_pop in enumerate(TARGET_POPULATIONS):
            NTWK['M_conn_%s_%s' % (afferent_pop, target_pop)] =\
                    build_fixed_aff_to_pop_matrix(afferent_pop, target_pop,\
                                                  NTWK['Model'], SEED=(SEED+1)*i+j+i*j)


def random_connections(NTWK, store_connections=False):
    
    CONN = np.empty((len(NTWK['POPS']), len(NTWK['POPS'])), dtype=object)
    CONN2 = []

    if store_connections:
        NTWK['connections'] = np.empty((len(NTWK['POPS']), len(NTWK['POPS'])), dtype=object)
    for ii, jj in itertools.product(range(len(NTWK['POPS'])), range(len(NTWK['POPS']))):
        if (NTWK['M'][ii,jj]['pconn']>0) and (NTWK['M'][ii,jj]['Q']!=0):
            if ('psyn' in NTWK['M'][ii,jj]) and (NTWK['M'][ii,jj]['psyn']<1):
                on_pre = 'G%s_post+=(rand()<%.3f)*w' % (NTWK['M'][ii,jj]['name'], NTWK['M'][ii,jj]['psyn']) # proba of release
            else:
                on_pre = 'G%s_post+=w' % NTWK['M'][ii,jj]['name']
                
            CONN[ii,jj] = brian2.Synapses(NTWK['POPS'][ii], NTWK['POPS'][jj], model='w:siemens', on_pre=on_pre)
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
            if store_connections:
                NTWK['connections'][ii,jj] = {'i':i_rdms, 'j':j_fixed}

    return CONN2

def draw_spatially_dependent_connectivity_profile(index_target_cell,
                                                  pool_of_potential_inputs_in_index_units,
                                                  number_of_desired_synapses,
                                                  spatial_decay_in_index_units,
                                                  delay_in_ms_per_index_units,
                                                  exclude_self=False):
    """
    ring geometry
    """
    Nmax = np.max(pool_of_potential_inputs_in_index_units)
    inputs = pool_of_potential_inputs_in_index_units
    
    input_to_target_distance = np.min([(inputs-index_target_cell)%Nmax, (index_target_cell-inputs)%Nmax], axis=0)
    delays = delay_in_ms_per_index_units*input_to_target_distance
    
    gaussian_proba = np.exp(-(input_to_target_distance/spatial_decay_in_index_units)**2)
    if exclude_self:
        gaussian_proba[index_target_cell] = 0.

    ipre = np.random.choice(inputs, number_of_desired_synapses, p=gaussian_proba/gaussian_proba.sum(), replace=False)
    return ipre, delays[ipre]
    
    
def random_distance_dependent_connections(NTWK):
    
    CONN = np.empty((len(NTWK['POPS']), len(NTWK['POPS'])), dtype=object)
    CONN2 = []

    for ii, jj in itertools.product(range(len(NTWK['POPS'])), range(len(NTWK['POPS']))):
        if (NTWK['M'][ii,jj]['pconn']>0) and (NTWK['M'][ii,jj]['Q']!=0):
            if ('psyn' in NTWK['M'][ii,jj]) and (NTWK['M'][ii,jj]['psyn']<1):
                on_pre = 'G%s_post+=(rand()<%.3f)*w' % (NTWK['M'][ii,jj]['name'], NTWK['M'][ii,jj]['psyn'])
            else:
                on_pre = 'G%s_post+=w' % NTWK['M'][ii,jj]['name']
                
            CONN[ii,jj] = brian2.Synapses(NTWK['POPS'][ii], NTWK['POPS'][jj], model='w:siemens', on_pre=on_pre)
            # CONN[ii,jj].connect(p=NTWK['M'][ii,jj]['pconn'], condition='i!=j')
            # N.B. the brian2 settings does weird things (e.g. it creates synchrony)
            # so we draw manually the connection to fix synaptic numbers
            N_per_cell = int(NTWK['M'][ii,jj]['pconn']*NTWK['POPS'][ii].N)
            if ii==jj: # need to take care of no autapse
                exclude_self = True
            else:
                exclude_self = False
            I_rdms, Delays = np.empty(0, dtype=int), np.empty(0)
            for index_target_cell in range(NTWK['POPS'][jj].N):
                i_rdms, delays = draw_spatially_dependent_connectivity_profile(index_target_cell,
                                                                               np.arange(NTWK['POPS'][ii].N),
                                                                               N_per_cell,
                                                                               NTWK['M'][ii,jj]['SpatialDecay'],
                                                                               NTWK['M'][ii,jj]['Delay'],
                                                                               exclude_self=exclude_self)
                I_rdms = np.concatenate([I_rdms, i_rdms])
                Delays = np.concatenate([Delays, delays])

            j_fixed = np.concatenate([np.ones(N_per_cell,dtype=int)*jjj for jjj in range(NTWK['POPS'][jj].N)])
            CONN[ii,jj].connect(i=np.array(I_rdms, dtype=int), j=j_fixed) 
            CONN[ii,jj].w = NTWK['M'][ii,jj]['Q']*brian2.nS
            CONN[ii,jj].delay = Delays*brian2.ms
            CONN2.append(CONN[ii,jj])

    return CONN2


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
        # reveral potential first
        if ('Erev_%s' % source_pop) in Model:
            Erev = Model['Erev_%s' % source_pop]
        elif len(source_pop.split('Exc'))>1:
            Erev = Model['Erev_Exc']
        elif len(source_pop.split('Inh'))>1:
            Erev = Model['Erev_Inh']
        else:
            print(' /!\ AFFERENT POP COULD NOT BE CLASSIFIED AS Exc or Inh /!\ ')
            print('-----> set to Exc by default')
            Erev= Model['Erev_Exc']
        # synaptic time course
        if ('Tsyn_%s' % source_pop) in Model:
            Tsyn = Model['Tsyn_%s' % source_pop]
        elif len(source_pop.split('Exc'))>1:
            Tsyn = Model['Tsyn_Exc']
        elif len(source_pop.split('Inh'))>1:
            Tsyn = Model['Tsyn_Inh']
        else:
            print(' /!\ AFFERENT POP COULD NOT BE CLASSIFIED AS Exc or Inh /!\ ')
            print('-----> set to Exc by default')
            Tsyn = Model['Tsyn_Exc']

        pconn, Qsyn, psyn = 0., 0., 1. # by default
        if ('p_'+source_pop+'_'+target_pop in Model) and ('Q_'+source_pop+'_'+target_pop in Model):
            pconn, Qsyn = Model['p_'+source_pop+'_'+target_pop], Model['Q_'+source_pop+'_'+target_pop]
        if ('psyn_'+source_pop+'_'+target_pop in Model):
            psyn = Model['psyn_'+source_pop+'_'+target_pop] # probability of release

        if (pconn==0) and verbose:
            print('No connection for:', source_pop,'->', target_pop)
                
        M[i, j] = {'pconn': pconn, 'Q': Qsyn,
                   'Erev': Erev, 'Tsyn': Tsyn, 'psyn':psyn,
                   'name':source_pop+target_pop}

        # in case conductance-current mixture
        if ('alpha_'+source_pop+'_'+target_pop in Model) and ('V0' in Model):
            M[i,j]['alpha'], M[i,j]['V0'] = Model['alpha_'+source_pop+'_'+target_pop], Model['V0']

        # in case of spatially decaying connectivity probability
        if ('SpatialDecay_'+source_pop+'_'+target_pop in Model):
            M[i,j]['SpatialDecay'] = Model['SpatialDecay_'+source_pop+'_'+target_pop]
            if ('Delay_'+source_pop+'_'+target_pop in Model): # with delays if present
                M[i,j]['Delay'] = Model['Delay_'+source_pop+'_'+target_pop]
            else:
                M[i,j]['Delay'] = 0
            
    if SI_units:
        print('synaptic network parameters in SI units')
        for m in M.flatten():
            m['Q'] *= 1e-9
            m['Erev'] *= 1e-3
            m['Tsyn'] *= 1e-3
            if 'V0' in m:
                m['V0'] *= 1e-3
    else:
        if verbose:
            print('synaptic network parameters --NOT-- in SI units')

    return M


def build_populations(Model, POPULATIONS,
                      AFFERENT_POPULATIONS=[],
                      with_raster=False,
                      with_pop_act=False,
                      with_Vm=0,
                      with_synaptic_currents=False,
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
            'AFFERENT_POPULATIONS':np.array(AFFERENT_POPULATIONS),
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
            
