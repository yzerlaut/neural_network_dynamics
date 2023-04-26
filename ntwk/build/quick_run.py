import numpy as np

from .syn_and_connec_construct import *
from ..stim.connect_afferent_input import *
from ..stim import waveform_library as stim_waveforms
from ..recording.load_and_save import write_as_hdf5
from ..theory import mean_field


def simulation(Model,
               filename='data.ntwk.h5', verbose=True, SEED=1):

    np.random.seed(SEED)
    ######################
    ## ----- Run  ----- ##
    ######################

    REC_POPS, AFF_POPS = list(Model['REC_POPS']), list(Model['AFF_POPS'])

    if verbose:
        print('--initializing simulation for %s [...]' % filename)

    #######################################
    ########### BUILD POPS ################
    #######################################

    NTWK = build_populations(Model, REC_POPS,
                                  AFFERENT_POPULATIONS=AFF_POPS,
                                  with_pop_act=True,
                                  with_raster=True,
                                  with_Vm=4)

    build_up_recurrent_connections(NTWK, SEED=5, verbose=verbose)

    #######################################
    ########### AFFERENT INPUTS ###########
    #######################################


    for a, aff_pop in enumerate(AFF_POPS):

        t_array = np.arange(int(Model['tstop']/Model['dt']))*Model['dt']
        if '%s_IncreasingStep_size' % aff_pop in Model:
            if verbose:
                print('Adding Increasing Step Waveform to:', aff_pop)
            faff =  stim_waveforms.IncreasingSteps(t_array, aff_pop, Model, translate_to_SI=False)
        elif 'F_%s' % aff_pop in Model:
            if verbose:
                print('Setting Constant Level to:', aff_pop)
            faff = Model['F_%s' % aff_pop]+0.*t_array
        else:
            print('/!\ no F_%s value set in Model ---> set to 0 ! /!\ ' % aff_pop)
            faff = Model['F_%s' % aff_pop]+0*t_array

        
        #  time-dep afferent excitation
        for i, tpop in enumerate(REC_POPS): # both on excitation and inhibition
            construct_feedforward_input(NTWK, tpop, aff_pop,
                                             t_array, faff,
                                             verbose=verbose,
                                             SEED=SEED+a+i+3)

    ################################################################
    ## --------------- Initial Condition ------------------------ ##
    ################################################################
    initialize_to_rest(NTWK)

    #####################
    ## ----- Run ----- ##
    #####################
    if verbose:
        print('----   running simulation for %s [...]' % filename)
    network_sim = collect_and_run(NTWK, verbose=verbose)

    #####################
    ## ----- Save ----- ##
    #####################
    write_as_hdf5(NTWK, filename=filename)
    if verbose:
        print('[ok] Results of the simulation are stored as:', filename)


        
def mean_field(Model, filename='data.mf.npz', verbose=True, dt=1e-2):
    """
    
    """
    if verbose:
        print('--   running (slow) mean-field for %s [...]' % filename)
        
    tstop = 1e-3*Model['tstop'] # from ms to seconds
    t = np.arange(int(tstop/dt))*dt

    DYN_SYSTEM, INPUTS, CURRENT_INPUTS = {}, {}, {}
    for rec in list(Model['REC_POPS']):

        DYN_SYSTEM[rec] = {'aff_pops':list(Model['AFF_POPS']),
                           'x0':1e-2}

        ### Intrinsic props
        if '%s_Ioscill_amp' % rec in Model:
            CURRENT_INPUTS[rec] = Model['%s_Ioscill_amp' % rec]*(1-np.cos(Model['%s_Ioscill_freq' % rec]*2*np.pi*t))/2.
        
        ### afferent input
        for aff in list(Model['AFF_POPS']):
            
            if '%s_IncreasingStep_size' % aff in Model:
                faff =  stim_waveforms.IncreasingSteps(t, aff, Model,
                                                       translate_to_SI=True)
            elif 'F_%s' % aff in Model:
                faff = Model['F_%s' % aff]+0.*t
            else:
                faff = Model['F_%s' % aff]+0*t

            INPUTS['%s_%s' % (aff,rec)] = faff
    
    X = mean_field.solve_mean_field_first_order(Model,
                                                DYN_SYSTEM,
                                                INPUTS=INPUTS,
                                                CURRENT_INPUTS=CURRENT_INPUTS,
                                                dt=dt, tstop=tstop)

    np.savez(filename, **X)

    
if __name__=='__main__':

    import sys
    sys.path.append('../configs/Network_Modulation_2020/')
    from model import Model
    for rec in list(Model['REC_POPS']):
        Model['COEFFS_%s' % rec] = np.load('configs/Network_Modulation_2020/COEFFS_pyrExc.npy')
    # quick_ntwk_sim(Model)
    quick_MF_sim(Model)

