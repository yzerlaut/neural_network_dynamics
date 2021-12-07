import os, sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import itertools

import main as ntwk

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

#####################################################
######## generate a transfer function's data ########
#####################################################

def generate_transfer_function(Model,\
                               scale='log'):
    """ Generate the data for the transfer function  """

    data = {'Fout_mean':[], 'Fout_std':[], 'Model':Model}
    
    SET_OF_FREQ_RANGES = []
    for pop in Model['POP_STIM']:
        data['F_%s' % pop] = []
        if 'F_%s_array' % pop in Model:
            SET_OF_FREQ_RANGES.append(Model["F_%s_array" % pop])
        else:
            print('need to set the range of scanned input by passing the "F_%s_array" in "Model"' % pop)
            
    print('============================================')
    print('             Starting Scan')
    print('============================================')
    for set_of_freqs in itertools.product(*SET_OF_FREQ_RANGES):

        Model['RATES'] = {}
        string_monitoring = '--> '
        for f, pop in zip(set_of_freqs, Model['POP_STIM']):
            Model['RATES']['F_%s'%pop] = f
            string_monitoring += '%s: %.1f, ' % (pop, f)
        print(string_monitoring)

        Fout_mean, Fout_std = run_sim(Model, firing_rate_only=True)
        # adding the input data
        for f, pop in zip(set_of_freqs, Model['POP_STIM']):
            data['F_%s' % pop].append(f)
        # adding the output data
        for f, key in zip([Fout_mean, Fout_std], ['Fout_mean', 'Fout_std']):
            data[key].append(f)
            
    # translating to 1d numpy array
    for key in ['F_%s' % pop for pop in Model['POP_STIM']]+['Fout_mean', 'Fout_std']:
        data[key] = np.array(data[key])
        
    print('============================================')
    print('             Scan finished')
    np.save(Model['filename'], data)
    print('Data saved as:', Model['filename'])
    print('============================================')
    print('---------------------------------------')
    return data
    
if __name__=='__main__':

    from neural_network_dynamics.transfer_functions.plots import *
    
    # import the model defined in root directory
    sys.path.append(os.path.join(\
                                 str(pathlib.Path(__file__).resolve().parents[1]),
                                 'configs', 'The_Spectrum_of_Asynch_Dyn_2018'))
    from model import *
    
    if sys.argv[-1]=='tf':
        Model['filename'] = 'data.npy'
        Model['NRN_KEY'] = 'RecExc' # we scan this population
        Model['N_SEED'] = 2 # seed repetition
        Model['POP_STIM'] = ['RecExc', 'RecInh']
        Model['F_RecExc_array'] = np.array([1, 2, 5, 10.])
        Model['F_RecInh_array'] = np.array([0.1, 1., 10.])
        generate_transfer_function(Model)
    elif sys.argv[-1]=='plot-tf':
        data = np.load('data.npy', allow_pickle=True).item()
        make_tf_plot_2_variables(data)
        ntwk.show()
    else:
        Model['NRN_KEY'] = 'RecExc' # we scan this population
        Model['N_SEED'] = 2 # seed repetition
        Model['POP_STIM'] = ['RecExc', 'RecInh']
        data = run_sim(Model, with_Vm=2, with_synaptic_currents=True)
        fig = plot_single_cell_sim(data)
        ntwk.show()
    
    ### TO TEST THE POLYNOM APPROX TO FIND THE RIGHT DOMAIN
    # Finput_previous, Fout_desired = np.array([0.1,0.2,0.5,1,5,10]), 1.2
    # Fout_previous = np.exp(Finput_previous/10)
    # find_right_input_value(neuron_params, SYN_POPS, RATES,
    #                        Finput_previous, Fout_previous, Fout_desired, with_plot=True)

    ### TO TEST THE TRANSFER FUNCTION GENERATION
