import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import main as ntwk
import numpy as np

def my_logspace(x1, x2, n):
    return np.logspace(np.log(x1)/np.log(10), np.log(x2)/np.log(10), n)

def built_up_neuron_params(Model):

    params = {'name':Model['NRN_KEY'], 'N':1}
    keys = ['Gl', 'Cm','Trefrac', 'El', 'Vthre', 'Vreset',\
            'delta_v', 'a', 'b', 'tauw']
    for k in keys:
        params[k] = Model[Model['NRN_KEY']+'_'+k]
    return params

def from_model_to_numerical_params(Model):

    if 'SYN_POPS' in Model.keys(): 
        SYN_POPS = Model['SYN_POPS'] # forced 
    else:
        SYN_POPS = []
        if 'RecExc' in Model['POP_STIM']:
            SYN_POPS.append({'name':'RecExc', 'Erev': Model['Ee'], 'N': Model['Ne'], 'Q': Model['Qee'], 'Tsyn': Model['Tse'], 'pconn': Model['pconn']})
        if 'AffExc' in Model['POP_STIM']:
            SYN_POPS.append({'name':'AffExc', 'Erev': Model['Ee'], 'N': Model['Na'], 'Q': Model['Qa'], 'Tsyn': Model['Tse'], 'pconn': Model['pconn_aff']})
        if 'RecInh' in Model['POP_STIM']:
            SYN_POPS.append({'name':'RecInh', 'Erev': Model['Ei'], 'N': Model['Ni'], 'Q': Model['Qie'], 'Tsyn':Model['Tsi'], 'pconn': Model['pconn']})
        if 'DsInh' in Model['POP_STIM']:
            SYN_POPS.append({'name':'DsInh', 'Erev': Model['Ei'], 'N': Model['Nd'], 'Q': Model['Qd'], 'Tsyn':Model['Tsi'], 'pconn': Model['pconn_dsnh']})

    RATES = {}
    if 'RATES' in Model.keys():
        RATES = Model['RATES']
    # elif (not Model['TF']) and (not 'RATES' in Model.keys()):
    else:
        for i, k in enumerate(Model['POP_STIM']):
            RATES['F_'+k] = Model['POP_RATES'][i]

    neuron_params = built_up_neuron_params(Model)

    return neuron_params, SYN_POPS, RATES

def run_sim(Model,
            with_Vm=0, with_synaptic_currents=False,
            firing_rate_only=False, tdiscard=100):

    neuron_params, SYN_POPS, RATES = from_model_to_numerical_params(Model)
    tstop, dt, SEED = Model['tstop'], Model['dt'], Model['SEED']

    if tdiscard>=tstop:
        print('discard time higher than simulation time -> set to 0')
        tdiscard = 0
        
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

def run_multiple_sim(Model, n_SEED=3):
                                
    vec = np.zeros(n_SEED)
    for seed in range(n_SEED):
        Model['SEED'] = Model['SEED']+seed
        vec[seed]= run_sim(Model, firing_rate_only=True)
    return vec.mean(), vec.std()


def find_right_input_value(Model,
                           Finput_previous, Fout_previous, Fout_desired,
                           with_plot=False):
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
    
def get_spiking_within_interval(Model,
                                scale='log'):

    """
    This functions calculates the firing rate response when the frequency of a given key increases (e.g. RecExc, the recurrent excitation)

    Note that, we also adjust the frequency of the input to have it spanning a given range (Fout_values)
    """

    Fout_min, Fout_max = Model['Fout_min'], Model['Fout_max']
    Finput_max, Finput_min = Model['Finput_min'], Model['Finput_max']
    N_input=Model['N_input']
    verbose=Model['verbose']
    
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
        Model['RATES']['F_'+Model['NRN_KEY']] = f # we set the input according to its current values
        OUTPUT_MEAN[i], OUTPUT_STD[i] = run_multiple_sim(Model)

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
        Finput_max = find_right_input_value(Model,
                                            INPUT, OUTPUT_MEAN, Fout_max)
        redo_scan = True
    elif (OUTPUT_MEAN[0]<Fout_min):
        if verbose: print('we need toto increase the minimum amplitude')
        Finput_min = find_right_input_value(Model,
                                            INPUT, OUTPUT_MEAN, Fout_min)
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
            Model['RATES']['F_'+Model['NRN_KEY']] = f # we set the input according to its current values
            OUTPUT_MEAN[i], OUTPUT_STD[i] = run_multiple_sim(Model)
        
    return INPUT, OUTPUT_MEAN, OUTPUT_STD
    
### generate a transfer function's data

def generate_transfer_function(Model,\
                               scale='log'):
    """ Generate the data for the transfer function  """

    N_input=Model['N_input']
    Finh, Faff, Fdsnh = Model['Finh_array'], Model['Faff_array'], Model['Fdsnh_array']
    print('============================================')
    print('             Starting Scan')
    print('============================================')
    if len(Fdsnh)>0:
        Fe, Fi, Fa, Fd = np.meshgrid(np.zeros(N_input), Finh, Faff, Fdsnh, indexing='ij')
        Fout_mean, Fout_std = 0*Fe, 0*Fe
        for i, fi in enumerate(Finh):
            print('--> inhibitory level', i, ' over ', len(Finh))
            for a, fa in enumerate(Faff):
                print('--> afferent level', a, ' over ', len(Faff))
                for d, fd in enumerate(Fdsnh):
                    print('--> dsnh level', d, ' over ', len(Fdsnh))
                    Model['RATES'] = {'F_RecExc':0.,'F_AffExc':fa, 'F_RecInh':fi, 'F_DsInh':fd}
                    Fe[:,i,a,d],\
                        Fout_mean[:,i,a,d],\
                        Fout_std[:,i,a,d] = get_spiking_within_interval(Model)
        data = {'Fe':Fe, 'Fi':Fi,
                'Fout_mean':Fout_mean, 'Fout_std':Fout_std,
                'cell_params':cell_params,'SYN_POPS':SYN_POPS}
    else:
        Fe, Fi, Fa = np.meshgrid(np.zeros(N_input), Finh, Faff, indexing='ij')
        Fout_mean, Fout_std = 0*Fe, 0*Fe
        for i, fi in enumerate(Finh):
            print('--> inhibitory level', i, ' over ', len(Finh))
            for a, fa in enumerate(Faff):
                print('--> afferent level', a, ' over ', len(Faff))
                Model['RATES'] = {'F_RecExc':0.,'F_AffExc':fa, 'F_RecInh':fi}
                Fe[:,i,a],\
                    Fout_mean[:,i,a],\
                    Fout_std[:,i,a] = get_spiking_within_interval(Model)
                
        data = {'Fe':Fe, 'Fi':Fi, 'Fa':Fa,
                'Fout_mean':Fout_mean, 'Fout_std':Fout_std,
                'cell_params':cell_params,'SYN_POPS':SYN_POPS, 'Model', Model}
    print('============================================')
    print('             Scan finished')
    np.save(Model['filename'], data)
    print('Data saved as:', Model['filename'])
    print('============================================')
    print('---------------------------------------')
    return data
    
if __name__=='__main__':

    # import the model defined in root directory
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from model import *
    # common to all protocols
    parser.add_argument('--POP_STIM', nargs='+', help='Set of desired populations', type=str, default=['RecExc', 'RecInh'])    
    parser.add_argument('--NRN_KEY', help='Neuron to stimulate', type=str, default='RecExc')    

    ### ==============================================================
    # type of stimulation
    ### ==============================================================
    # Single Stimulation by default, just need to set the input rates:
    parser.add_argument('--POP_RATES', nargs='+', help='Set of rates of the populations', type=float, default=[1, 1])    
    # TRANSFER FUNCTION (params scan)
    parser.add_argument('--TF', help="Run the transfer function", action="store_true") # SET TO TRUE TO RUN TF
    parser.add_argument('--N_input', help='discretization of input', type=int, default=4)    
    parser.add_argument('--n_SEED', help='number of varied seed', type=int, default=2)    
    parser.add_argument('--Fout_min', help='min output firing rate', type=float, default=1e-2)    
    parser.add_argument('--Fout_max', help='max output firing rate', type=float, default=30.)    
    parser.add_argument('--Finput_min', help='min input firing rate (of varied population)', type=float, default=1e-2)    
    parser.add_argument('--Finput_max', help='max input firing rate (of varied population)', type=float, default=30.)
    # now range for inputs
    parser.add_argument('--Finh_array', nargs='+', help='Inhibitory firing rates', type=float, default=my_logspace(1e-2, 10, 3))    
    parser.add_argument('--Faff_array', nargs='+', help='Afferent firing rates', type=float, default=my_logspace(4, 20, 3))    
    parser.add_argument('--Fdsnh_array', nargs='+', help='DisInhibitory firing rates', type=float, default=[])    
    
    # additional stuff
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("--filename", '-f', help="filename",type=str, default='data.npy')

    args = parser.parse_args()

    Model = vars(args)

    if args.TF:
        generate_transfer_function(Model)
    else:
        from neural_network_dynamics.transfer_functions.plots import *
        data = run_sim(Model, with_Vm=1, with_synaptic_currents=True);
        plot_single_cell_sim(data)
        plt.show()
    
                                

    ### TO TEST THE POLYNOM APPROX TO FIND THE RIGHT DOMAIN
    # Finput_previous, Fout_desired = np.array([0.1,0.2,0.5,1,5,10]), 1.2
    # Fout_previous = np.exp(Finput_previous/10)
    # find_right_input_value(neuron_params, SYN_POPS, RATES,
    #                        Finput_previous, Fout_previous, Fout_desired, with_plot=True)

    ### TO TEST THE TRANSFER FUNCTION GENERATION
