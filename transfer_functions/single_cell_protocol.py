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
            with_Vm=0, with_synaptic_currents=False,
            firing_rate_only=False, tdiscard=100):

    if tdiscard>=tstop:
        print('discard time higher than simulation time -> set to 0')
        tdiscard = 0
        
    neuron_params = add_other_necessary_keys(neuron_params)
    t_array = np.arange(int(tstop/dt))*dt

    NEURONS = [{'name':'Target', 'N':1, 'type':'', 'params':neuron_params}]

    ############################################################################
    # everything is reformatted to have it compatible with the network framework
    ############################################################################
    
    M = []
    for syn in SYN_POPS:
        M.append([{'Q': 0., 'Erev': syn['Erev'], 'Tsyn': syn['Tsyn'],
                   'name': syn['name']+NEURONS[0]['name'], 'pconn': 0.}])
    M = np.array(M)    

    VMS, ISYNe, ISYNi = [], [], [] # initialize to empty
    if with_Vm and with_synaptic_currents:
        NTWK = ntwk.build_populations(NEURONS, M,
                                      with_Vm=with_Vm, with_raster=True,
                                      with_synaptic_currents=with_synaptic_currents)
    elif with_Vm:
        NTWK = ntwk.build_populations(NEURONS, M, with_Vm=with_Vm, with_raster=True)
    else:
        NTWK = ntwk.build_populations(NEURONS, M, with_raster=True)

    ntwk.initialize_to_rest(NTWK) # (fully quiescent State as initial conditions)

    SPKS, SYNAPSES, PRESPKS = [], [], []

    for i, syn in enumerate(SYN_POPS):
        afferent_pop = {'Q':syn['Q'], 'N':syn['N'], 'pconn':syn['pconn']}
        rate_array = RATES['F_'+syn['name']]+0.*t_array
        ntwk.construct_feedforward_input(NTWK, NTWK['POPS'][0],
                                         afferent_pop, t_array, rate_array,
                                         conductanceID=syn['name']+NEURONS[0]['name'],
                                         SEED=i+SEED, with_presynaptic_spikes=True)

    sim = ntwk.collect_and_run(NTWK, tstop=tstop, dt=dt)

    if firing_rate_only:
        tspikes = np.array(NTWK['RASTER'][0].t/ntwk.ms)
        return 1e3*len(tspikes[tspikes>tdiscard])/(tstop-tdiscard) # from ms to Hz
    else:
        output = {'ispikes':np.array(NTWK['RASTER'][0].i), 'tspikes':np.array(NTWK['RASTER'][0].t/ntwk.ms),
                  'dt':str(dt), 'tstop':str(tstop), 'SYN_POPS':SYN_POPS, 'params':neuron_params}

        if with_Vm:
            output['i_prespikes'] = NTWK['iRASTER_PRE']
            output['t_prespikes'] = [vv/ntwk.ms for vv in NTWK['tRASTER_PRE']]
            output['Vm'] = np.array([vv.V/ntwk.mV for vv in NTWK['VMS'][0]])
        if with_synaptic_currents:
            output['Ie'] = np.array([vv.Ie/ntwk.pA for vv in NTWK['ISYNe'][0]])
            output['Ii'] = np.array([vv.Ii/ntwk.pA for vv in NTWK['ISYNi'][0]])
        return output

def run_multiple_sim(cell_params, SYN_POPS, RATES,
                     SEED=3, n_SEED=3, dt=0.1, tstop=200):
                                
    vec = np.zeros(n_SEED)
    for seed in range(n_SEED):
        vec[seed]= run_sim(cell_params, SYN_POPS, RATES,
                           tstop=tstop, dt=dt, SEED=SEED+seed,
                           firing_rate_only=True)

    return vec.mean(), vec.std()


def find_right_input_value(cell_params, SYN_POPS, RATES,
                           Finput_previous, Fout_previous, Fout_desired,
                           key_to_vary='RecExc', with_plot=False):
    """
    A kind of newton method to find the right input leading to fout_desired
    given the previous observations
    """

    to_min = Fout_desired-Fout_previous
    i0 = min([len(Fout_previous)-3, max([0,len(to_min[to_min>=0])-1])])
    pol = np.polyfit(Finput_previous[i0:i0+3], Fout_previous[i0:i0+3], 3)
    Fin, Fout = Finput_previous[i0:i0+3], Fout_previous[i0:i0+3]
    Fin2, Fout2 = np.linspace(Fin.min(), Fin.max()), np.linspace(Fout.min(), Fout.max())
    new_Finput =  Fin2[np.argmin(np.abs(np.polyval(pol, Fin2)-Fout_desired))]

    if with_plot:
        plt.plot(Finput_previous, Fout_previous, 'o')
        plt.plot(Fin, Fout, 'o')
        plt.plot(Fin2, np.polyval(pol, Fin2), '-')
        plt.plot([new_Finput], [Fout_desired], 'o')
        plt.show()
        
    if new_Finput>0:    
        return new_Finput
    else:
        return 0
    
def get_spiking_within_interval(cell_params, SYN_POPS, RATES,
                                Fout_min = 1e-2,
                                Fout_max = 40,
                                Finput_max = 20.,
                                Finput_min = 1e-2,
                                N_input=3,
                                key_to_vary='RecExc',
                                scale='log',
                                SEED=3, n_SEED=3, dt=0.1, tstop=200,
                                verbose=False, inf_loop_security=500000):

    """
    This functions calculates the firing rate response when the frequency of a given key increases (e.g. RecExc, the recurrent excitation)

    Note that, we also adjust the frequency of the input to have it spanning a given range (Fout_values)
    """
    
    #
    if scale=='log':
        if Finput_min==0:
            print('min input can not be zero in log mode, set to 1e-3')
            Finput_min = 1e-3
        INPUT = np.logspace(np.log(Finput_min)/np.log(10), np.log(Finput_max)/np.log(10), N_input)
    if scale=='lin':
        INPUT = np.linspace(Finput_min, Finput_max, N_input)
    OUTPUT_MEAN, OUTPUT_STD = np.zeros(INPUT.size), np.zeros(INPUT.size)
    
    ### ==================================================
    ### let's do a first scan
    ### ==================================================
    for i, f in enumerate(INPUT):
        RATES['F_'+key_to_vary] = f # we set the input according to its current values
        OUTPUT_MEAN[i], OUTPUT_STD[i] = run_multiple_sim(cell_params, SYN_POPS, RATES,
                                                         tstop=tstop, dt=dt, SEED=SEED)

    ### =============================================================================
    ### Now case by case analysis to have the output firing rate in the range we want
    ### =============================================================================
    redo_scan = False # by default
    if (OUTPUT_MEAN[0]>=Fout_min) and (OUTPUT_MEAN[-1]<=Fout_max):
        if verbose: print('fine this is what we want')
        pass
    elif (OUTPUT_MEAN[0]<Fout_min) and (OUTPUT_MEAN[-1]<Fout_min):
        if verbose: print('no hope to get the desired spiking in this range')
        pass
    elif (OUTPUT_MEAN[-1]>Fout_max):
        if verbose: print('we need to lower the maximum amplitude')
        Finput_max = find_right_input_value(cell_params, SYN_POPS, RATES,
                                            INPUT, OUTPUT_MEAN, Fout_max,
                                            key_to_vary=key_to_vary):
        redo_scan = True
    elif (OUTPUT_MEAN[0]<Fout_min):
        if verbose: print('we need toto increase the minimum amplitude')
        Finput_min = find_right_input_value(cell_params, SYN_POPS, RATES,
                                            INPUT, OUTPUT_MEAN, Fout_min,
                                            key_to_vary=key_to_vary):
        redo_scan = True
    else:
        if verbose: print('case not taken into account, we return the default scan')
        redo_scan = False

    if redo_scan:
        ## HERE WE REDO THE SCAN
        if scale=='log':
            if Finput_min==0:
                print('min input can not be zero in log mode, set to 1e-3')
                Finput_min = 1e-3
            INPUT = np.logspace(np.log(Finput_min)/np.log(10), np.log(Finput_max)/np.log(10), N_input)
        if scale=='lin':
            INPUT = np.linspace(Finput_min, Finput_max, N_input)
        for i, f in enumerate(INPUT):
            RATES['F_'+key_to_vary] = f # we set the input according to its current values
            OUTPUT_MEAN[i], OUTPUT_STD[i] = run_multiple_sim(cell_params, SYN_POPS, RATES,
                                                             tstop=tstop, dt=dt, SEED=SEED)
        
    # ### ==================================================
    # ### response at the min level we look at :
    # ### ==================================================
    # if verbose:
    #     print('firing rate at the baseline level:', OUTPUT_MEAN[0], '+/-', OUTPUT_STD[0], 'Hz')

    # ### ==================================================
    # ### response at the max level we look at :
    # ### ==================================================
    # RATES['F_'+key_to_vary] = INPUT[-1] # we set the input according to its current values
    # OUTPUT_MEAN[-1], OUTPUT_STD[-1] = run_multiple_sim(cell_params, SYN_POPS, RATES,
    #                                                  tstop=tstop, dt=dt, SEED=SEED)
    # if verbose:
    #     print('firing rate at the max level:', OUTPUT_MEAN[-1], '+/-', OUTPUT_STD[-1], 'Hz')

    # # we adjust the values to have it above this baseline level
    # Fout_values = np.cumsum(np.concatenate([[FINAL_OUTPUT_MEAN[0]+Fout_values[0]], np.diff(Fout_values)]))

    # i_input = 1
    # ntot = 0
    # while (i_input<len(Fout_values)) and ntot<inf_loop_security:
    #     vec = np.zeros(n_SEED)
    #     RATES['F_'+key_to_vary] = FINAL_INPUT[i_input+1] # we set the input according to its current values
    #     vec[0]= run_sim(cell_params, SYN_POPS, RATES,
    #                             tstop=tstop, dt=dt, SEED=SEED,
    #                             firing_rate_only=True)
    #     if (vec[0]>=Fout_values[i_input]): # if we make a too big jump
    #         # we redo it until the jump is ok (so by a small rescaling of fe)
    #         # we divide the step by 2
    #         FINAL_INPUT[i_input+1] = FINAL_INPUT[i_input]+(FINAL_INPUT[i_input+1]-FINAL_INPUT[i_input])/2.
    #         if verbose:
    #             print("we rescale the fe vector [...], to ", FINAL_INPUT[i_input+1],
    #                   '(because fout=', str(vec[0]), 'higher than ', Fout_values[i_input],')')
    #     else: # we can run the rest
    #         if verbose:
    #             print("== the excitation level :", i_input," over ",Finput.size)
    #         for seed in range(1,n_SEED):
    #             vec[seed]= run_sim(cell_params, SYN_POPS, RATES,
    #                                tstop=tstop, dt=dt,SEED=SEED+seed,
    #                                firing_rate_only=True)
    #             if verbose:
    #                 print("== ---- _____________ seed :",seed)
    #         FINAL_OUTPUT_MEAN[i_input+1] = vec.mean()
    #         FINAL_OUTPUT_STD[i_input+1] = vec.std()
    #         if verbose:
    #             print("== ---- ===> Fout :", FINAL_OUTPUT_MEAN[i_input+1])
    #         if i_input<FINAL_INPUT.size-2: # we set the next value to the next one...
    #             FINAL_INPUT[i_input+2] = FINAL_INPUT[i_input+1]+(Finput[i_input+1]-Finput[i_input])
    #         i_input += 1 # and we progress in the loop
            
    #     ntot+=1 # security for infinite loops...
    return INPUT, OUTPUT_MEAN, OUTPUT_STD
    
### generate a transfer function's data

def generate_transfer_function(cell_params, SYN_POPS,\
                               Fout_max = 40.,
                               F_exc_max = 20.,
                               F_exc_min = 1e-2,
                               Finh = np.logspace(-2, 1.7, 4),
                               Faff = np.logspace(0.6, 1.3, 4),
                               Fdsnh = None,
                               SEED=3, n_SEED=4,\
                               verbose=False,
                               filename='data/example_data.npy',
                               dt=0.1, tstop=1000):
    """ Generate the data for the transfer function  """

    if Fdsnh is not None:
        Fe, Fi, Fa, Fd = np.meshgrid(np.concatenate([[0],Fexc]), Finh, Faff, Fdsnh, indexing='ij')
        Fout_mean, Fout_std = 0*Fe, 0*Fe
        for i, fi in enumerate(Finh):
            for a, fa in enumerate(Faff):
                for d, fd in enumerate(Fdsnh):
                    RATES = {'F_RecExc':0.,'F_AffExc':fa, 'F_RecInh':fi, 'F_DsInh':fd}
                    Fe[:,i,a,d],\
                        Fout_mean[:,i,a,d],\
                        Fout_std[:,i,a,d] = get_spiking_within_interval(cell_params, SYN_POPS, RATES,
                                                                        Fout_values = Fout_values,
                                                                        Finput = Fexc,
                                                                        SEED=SEED, n_SEED=n_SEED,
                                                                        dt=dt, tstop=tstop,
                                                                        verbose=verbose)
        np.save(filename, [Fe, Fi, Fa, Fd, Fout_mean, Fout_std, cell_params, SYN_POPS, RATES])
        return Fe, Fi, Fa, Fd, Fout_mean, Fout_std
    else:
        Fe, Fi, Fa = np.meshgrid(np.concatenate([[0],Fexc]), Finh, Faff, indexing='ij')
        Fout_mean, Fout_std = 0*Fe, 0*Fe
        for i, fi in enumerate(Finh):
            for a, fa in enumerate(Faff):
                RATES = {'F_RecExc':0.,'F_AffExc':fa, 'F_RecInh':fi}
                Fe[:,i,a],\
                    Fout_mean[:,i,a],\
                    Fout_std[:,i,a] = get_spiking_within_interval(cell_params, SYN_POPS, RATES,
                                                                    Fout_values = Fout_values,
                                                                    Finput = Fexc,
                                                                    SEED=SEED, n_SEED=n_SEED,
                                                                    dt=dt, tstop=tstop,
                                                                    verbose=verbose)
        np.save(filename, [Fe, Fi, Fa, Fout_mean, Fout_std, cell_params, SYN_POPS, RATES])
        return Fe, Fi, Fa, Fout_mean, Fout_std
    
if __name__=='__main__':

    import matplotlib.pylab as plt
    neuron_params = {'N':1,\
                     'Gl':10., 'Cm':150.,'Trefrac':5.,\
                     'El':-65., 'Vthre':-50., 'Vreset':-65.}
    
    SYN_POPS = [{'name':'RecExc', 'Erev': 0, 'N': 4000, 'Q': 2, 'Tsyn': 5., 'pconn': 0.05},
                {'name':'AffExc', 'Erev': 0, 'N': 100, 'Q': 5, 'Tsyn': 5., 'pconn': 0.1},
                {'name':'RecInh', 'Erev': -80, 'N': 1000, 'Q': 10., 'Tsyn': 5., 'pconn': 0.05}]


    RATES = {'F_RecExc':1.,'F_AffExc':6., 'F_RecInh':1.}

    # data = run_sim(neuron_params, SYN_POPS, RATES,
    #             tstop=1000, with_Vm=True)
    # plt.plot(data['Vm'][0])
    # plt.show()

    # generate_transfer_function(neuron_params, SYN_POPS,
    # a,b,c = get_spiking_within_interval(neuron_params, SYN_POPS, RATES,
    #                             Fout_max = 40.,
    #                             Finput_min = 1e-2,
    #                             Finput_max = 20.,
    #                             N_input=3,
    #                             SEED=3, n_SEED=2,
    #                             tstop=1000, dt=0.5,
    #                             verbose=True)

    ### TO TEST THE POLYNOM APPROX TO FIND THE RIGHT DOMAIN
    # Finput_previous, Fout_desired = np.array([0.1,0.2,0.5,1,5,10]), 1.2
    # Fout_previous = np.exp(Finput_previous/10)
    # find_right_input_value(neuron_params, SYN_POPS, RATES,
    #                        Finput_previous, Fout_previous, Fout_desired, with_plot=True)

    
    # print(a,b,c)
