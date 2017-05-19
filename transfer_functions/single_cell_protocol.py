import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import main as ntwk
import numpy as np

def add_other_necessary_keys(params):
    params1 = params.copy()
    keys = ['Gl', 'Cm','Trefrac', 'El', 'Vthre', 'Vreset',\
            'delta_v', 'a', 'b', 'tauw', 'name', 'N']
    ## DEFAULT VALUES FOR THOSE PARAMETERS UNLESS OTHERWISE SPECIFIED !
    default = [10., 150., 5., -65., -50., -65.,\
               0., 0., 0., 1e9, '', 1]
    for k, d in zip(keys, default):
        if not k in params.keys(): # if not defined
            params1[k] = d # default value
    return params1

def run_sim(neuron_params, SYN_POPS, RATES, dt=0.1, tstop=100., SEED=1,
            with_Vm=0, with_synaptic_currents=False):

    neuron_params = add_other_necessary_keys(neuron_params)
    t_array = np.arange(int(tstop/dt))*dt

    NEURONS = [{'name':'Target', 'N':1, 'type':'', 'params':neuron_params}]

    ############################################################################
    # everything is reformatted to have it compatible with the network framework
    ############################################################################
    
    M = []
    for syn in SYN_POPS:
        M.append([{'Q': 0., 'Erev': syn['Erev'], 'Tsyn': syn['Tsyn'], 'name': syn['name']+NEURONS[0]['name'], 'pconn': 0.}])
    M = np.array(M)    

    VMS, ISYNe, ISYNi = [], [], [] # initialize to empty
    if with_Vm and with_synaptic_currents:
        NTWK = ntwk.build_populations(NEURONS, M,
                                      with_Vm=with_Vm, with_raster=True,
                                      with_synaptic_currents=with_synaptic_currents, verbose=True)
    elif with_Vm:
        NTWK = ntwk.build_populations(NEURONS, M, with_Vm=with_Vm, with_raster=True)
    else:
        NTWK = ntwk.build_populations(NEURONS, M, with_raster=True)

    ntwk.initialize_to_rest(NTWK) # (fully quiescent State as initial conditions)

    SPKS, SYNAPSES, PRESPKS = [], [], []

    for i, syn in enumerate(SYN_POPS):
        afferent_pop = {'Q':syn['Q'], 'N':syn['N'], 'pconn':syn['pconn']}
        rate_array = RATES[i]+0.*t_array
        ntwk.construct_feedforward_input(NTWK, NTWK['POPS'][0],
                                         afferent_pop, t_array, rate_array,
                                         conductanceID=syn['name']+NEURONS[0]['name'],
                                         SEED=i+SEED, with_presynaptic_spikes=True)

    sim = ntwk.collect_and_run(NTWK, tstop=tstop, dt=dt)

    output = {'ispikes':np.array(NTWK['RASTER'][0].i), 'tspikes':np.array(NTWK['RASTER'][0].t/ntwk.ms),
              'dt':str(dt), 'tstop':str(tstop)}
    
    if with_Vm:
        # output['i_prespikes'] = np.concatenate([i*np.ones(len(presk)) for i, presk in enumerate(PRESPKS)]).flatten()
        # output['t_prespikes'] = np.concatenate([presk/ntwk.ms for presk in PRESPKS]).flatten()
        output['Vm'] = np.array([vv.V/ntwk.mV for vv in NTWK['VMS'][0]])
    if with_synaptic_currents:
        output['Ie'] = np.array([vv.Ie/ntwk.pA for vv in NTWK['ISYNe'][0]])
        output['Ii'] = np.array([vv.Ii/ntwk.pA for vv in NTWK['ISYNi'][0]])

    return output
    
if __name__=='__main__':

    neuron_params = {'N':1,\
                     'Gl':10., 'Cm':150.,'Trefrac':5.,\
                     'El':-65., 'Vthre':-50., 'Vreset':-65.}
    RATES = [10.,10.,10.,10.]
    SYN_POPS = [{'name':'exc1', 'Erev': 0.0, 'N': 1000, 'Q': .3, 'Tsyn': 5., 'pconn': 0.1},
                {'name':'exc2', 'Erev': 0.0, 'N': 1000, 'Q': .5, 'Tsyn': 5., 'pconn': 0.1},
                {'name':'inh1', 'Erev': -80.0, 'N': 1000, 'Q': 10., 'Tsyn': 5., 'pconn': 0.1},
                {'name':'inh2', 'Erev': -80.0, 'N': 1000, 'Q': 4., 'Tsyn': 5., 'pconn': 0.1}]

    data = run_sim(neuron_params, SYN_POPS, RATES,
                tstop=1000, with_Vm=True)

    import matplotlib.pylab as plt
    plt.plot(data['Vm'][0])
    plt.show()
