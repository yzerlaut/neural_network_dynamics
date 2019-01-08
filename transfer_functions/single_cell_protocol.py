import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import main as ntwk
import numpy as np

def my_logspace(x1, x2, n):
    return np.logspace(np.log(x1)/np.log(10), np.log(x2)/np.log(10), n)

def run_sim(Model,
            with_Vm=0, with_synaptic_currents=False,
            firing_rate_only=False, tdiscard=100):

    if 'RATES' in Model.keys():
        RATES = Model['RATES']
    else:
        RATES = {}
        for pop in Model['POP_STIM']:
            RATES['F_'+pop] = Model['F_'+pop]

    if tdiscard>=Model['tstop']:
        print('discard time higher than simulation time -> set to 0')
        tdiscard = 0
        
    t_array = np.arange(int(Model['tstop']/Model['dt']))*Model['dt']

    ############################################################################
    # everything is reformatted to have it compatible with the network framework
    ############################################################################
    
    aff_pops_discard_self = []
    for p in Model['POP_STIM']:
        if p!=Model['NRN_KEY']:
            aff_pops_discard_self.append(p)

    # note that number of neurons become number of different seeds
    NTWK = ntwk.build_populations(Model, [Model['NRN_KEY']],
                                  NEURONS = [{'name':Model['NRN_KEY'], 'N':Model['N_SEED']}],
                                  AFFERENT_POPULATIONS=aff_pops_discard_self,
                                  with_Vm=with_Vm, with_raster=True,
                                  with_synaptic_currents=with_synaptic_currents)

    ntwk.initialize_to_rest(NTWK) # (fully quiescent State as initial conditions)

    SPKS, SYNAPSES, PRESPKS = [], [], []

    for i, afferent_pop in enumerate(Model['POP_STIM']):
        rate_array = RATES['F_'+afferent_pop]+0.*t_array
        ntwk.construct_feedforward_input(NTWK, Model['NRN_KEY'], afferent_pop,
                                         t_array, rate_array,
                                         SEED=i+Model['SEED'])

    sim = ntwk.collect_and_run(NTWK)

    # calculating firing rate
    vec = np.zeros(Model['N_SEED'])
    ispikes = np.array(NTWK['RASTER'][0].i)
    tspikes = np.array(NTWK['RASTER'][0].t/ntwk.ms)
    for nrn in range(Model['N_SEED']):
        i0 = np.argwhere(ispikes==nrn).flatten()
        ts = tspikes[i0]
        fout = 1e3*len(ts[ts>tdiscard])/(Model['tstop']-tdiscard) # from ms to Hz
        vec[nrn]= fout
        
    if firing_rate_only:
        return vec.mean(), vec.std()
    else:
        output = {'ispikes':np.array(NTWK['RASTER'][0].i),
                  'tspikes':np.array(NTWK['RASTER'][0].t/ntwk.ms),
                  'Model':Model, 'fout_mean':vec.mean(), 'fout_std':vec.std()}
        if with_Vm:
            output['i_prespikes'] = NTWK['iRASTER_PRE']
            output['t_prespikes'] = [vv/ntwk.ms for vv in NTWK['tRASTER_PRE']]
            output['Vm'] = np.array([vv.V/ntwk.mV for vv in NTWK['VMS'][0]])
        if with_synaptic_currents:
            output['Ie'] = np.array([vv.Ie/ntwk.pA for vv in NTWK['ISYNe'][0]])
            output['Ii'] = np.array([vv.Ii/ntwk.pA for vv in NTWK['ISYNi'][0]])
        return output

### generate a transfer function's data
import itertools
def generate_transfer_function(Model,\
                               scale='log'):
    """ Generate the data for the transfer function  """

    data = {'F_RecExc':[], 'F_RecInh':[], 'F_AffExc':[], 'F_DsInh':[],
            'Fout_mean':[], 'Fout_std':[], 'Model':Model}

    print('============================================')
    print('             Starting Scan')
    print('============================================')
    
    for fe, fi, fa, fd in itertools.product(Model['F_RecExc_array'], Model['F_RecInh_array'],\
                                            Model['F_AffExc_array'], Model['F_DsInh_array']):
        print('--> excitatory level:', fe, ' inhibitory level:', fi, ' afferent level', fa, ' dsnh level', fd)
        Model['RATES'] = {'F_RecExc':fe,'F_AffExc':fa, 'F_RecInh':fi, 'F_DsInh':fd}
        Fout_mean, Fout_std = run_sim(Model, firing_rate_only=True)
        # adding the data
        for f, key in zip([fe, fi, fa, fd, Fout_mean, Fout_std],\
                          ['F_RecExc', 'F_RecInh', 'F_AffExc', 'F_DsInh', 'Fout_mean', 'Fout_std']):
            data[key].append(f)
            
    # translating to 1d numpy array
    for key in ['F_RecInh', 'F_AffExc', 'F_DsInh', 'F_RecExc', 'Fout_mean', 'Fout_std']:
        data[key] = np.array(data[key])
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
    parser.add_argument('--POP_STIM', nargs='+', help='Set of desired populations', type=str,
                        default=['RecExc', 'RecInh'])    
    parser.add_argument('--NRN_KEY', help='Neuron to stimulate', type=str, default='RecExc')    

    ### ==============================================================
    # type of stimulation
    ### ==============================================================
    
    # TRANSFER FUNCTION (params scan)
    parser.add_argument('--TF', help="Run the transfer function", action="store_true") # SET TO TRUE TO RUN TF
    
    parser.add_argument('--N_SEED', help='number of varied seed', type=int, default=1)    
    # now range for inputs
    parser.add_argument('--F_RecExc_array',
                        nargs='+', help='Excitatory firing rates', type=float, default=my_logspace(1e-2, 10, 3))
    parser.add_argument('--F_RecInh_array',
                        nargs='+', help='Inhibitory firing rates', type=float, default=my_logspace(1e-2, 10, 3))
    parser.add_argument('--F_AffExc_array',
                        nargs='+', help='Afferent firing rates', type=float, default=my_logspace(4, 20, 3))    
    parser.add_argument('--F_DsInh_array',
                        nargs='+', help='DisInhibitory firing rates', type=float, default=[0])    
    # additional stuff
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("--filename", '-f', help="filename",type=str, default='data.npy')

    args = parser.parse_args()

    Model = vars(args)

    if args.TF:
        generate_transfer_function(Model)
    else:
        from neural_network_dynamics.transfer_functions.plots import *
        plt.style.use('ggplot')
        data = run_sim(Model, with_Vm=Model['N_SEED'], with_synaptic_currents=True)
        fig = plot_single_cell_sim(data)
        ntwk.show()
    
    ### TO TEST THE POLYNOM APPROX TO FIND THE RIGHT DOMAIN
    # Finput_previous, Fout_desired = np.array([0.1,0.2,0.5,1,5,10]), 1.2
    # Fout_previous = np.exp(Finput_previous/10)
    # find_right_input_value(neuron_params, SYN_POPS, RATES,
    #                        Finput_previous, Fout_previous, Fout_desired, with_plot=True)

    ### TO TEST THE TRANSFER FUNCTION GENERATION
