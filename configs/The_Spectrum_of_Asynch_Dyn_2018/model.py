Model = {
    ## ---------------------------------------------------------------------------------
    ### Initialisation by default parameters
    ## UNIT SYSTEM is : ms, mV, pF, nS, pA, Hz (arbitrary and unconsistent, so see code)
    ## ---------------------------------------------------------------------------------
    # numbers of neurons in population
    'N_RecExc':4000, 'N_RecInh':1000, 'N_DsInh':500, 'N_AffExc':100,
    # synaptic weights
    'Q_RecExc_RecExc':2., 'Q_RecExc_RecInh':2., 
    'Q_RecInh_RecExc':10., 'Q_RecInh_RecInh':10., 
    'Q_AffExc_RecExc':4., 'Q_AffExc_RecInh':4.,  'Q_AffExc_DsInh':4.,
    'Q_DsInh_RecInh':10., 
    # synaptic time constants
    'Tse':5., 'Tsi':5.,
    # synaptic reversal potentials
    'Ee':0., 'Ei': -80.,
    # connectivity parameters
    'p_RecExc_RecExc':0.05, 'p_RecExc_RecInh':0.05, 
    'p_RecInh_RecExc':0.05, 'p_RecInh_RecInh':0.05, 
    'p_AffExc_RecExc':0.1, 'p_AffExc_RecInh':0.1, 
    'p_AffExc_DsInh':0.075,
    'p_DsInh_RecInh':0.05, 
    # afferent stimulation
    'F_AffExc':20., 'F_DsInh':0.,
    # recurrent activity (for single cell simulation only)
    'F_RecExc':1., 'F_RecInh':1.,
    # simulation parameters
    'dt':0.1, 'tstop': 6000., 'SEED':3, # low by default, see later
    ## ---------------------------------------------------------------------------------
    # === cellular properties (based on AdExp), population by population ===
    # --> Excitatory population (RecExc, recurrent excitation)
    'RecExc_Gl':10., 'RecExc_Cm':200.,'RecExc_Trefrac':5.,
    'RecExc_El':-70., 'RecExc_Vthre':-50., 'RecExc_Vreset':-70., 'RecExc_deltaV':0.,
    'RecExc_a':0., 'RecExc_b': 0., 'RecExc_tauw':1e9,
    # --> Inhibitory population (RecInh, recurrent inhibition)
    'RecInh_Gl':10., 'RecInh_Cm':200.,'RecInh_Trefrac':5.,
    'RecInh_El':-70., 'RecInh_Vthre':-53., 'RecInh_Vreset':-70., 'RecInh_deltaV':0.,
    'RecInh_a':0., 'RecInh_b': 0., 'RecInh_tauw':1e9,
    # --> Disinhibitory population (DsInh, disinhibition)
    'DsInh_Gl':10., 'DsInh_Cm':200.,'DsInh_Trefrac':5.,
    'DsInh_El':-70., 'DsInh_Vthre':-50., 'DsInh_Vreset':-70., 'DsInh_deltaV':0.,
    'DsInh_a':0., 'DsInh_b': 0., 'DsInh_tauw':1e9
}

if __name__=='__main__':

    import sys
    sys.path.append('../..')
    import main as ntwk

    
    if sys.argv[-1]=='plot':
        ## load file
        data = ntwk.load_dict_from_hdf5('CellRep2019_data.h5')
        print(data)
        # ## plot
        fig, _ = ntwk.raster_and_Vm_plot(data, smooth_population_activity=10.)
        ntwk.show()
    
    else:
        import numpy as np
        from analyz.processing.signanalysis import smooth
        
        NTWK = ntwk.build_populations(Model, ['RecExc', 'RecInh', 'DsInh'],
                                      AFFERENT_POPULATIONS=['AffExc'],
                                      with_raster=True,
                                      with_Vm=4,
                                      verbose=True)

        ntwk.build_up_recurrent_connections(NTWK, SEED=5, verbose=True)

        #######################################
        ########### AFFERENT INPUTS ###########
        #######################################


        t_array = ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']
        faff = smooth(np.array([4*int(tt/1000) for tt in t_array]), int(200/0.1))       

        # ######################
        # ## ----- Plot ----- ##
        # ######################

        # # # afferent excitation onto cortical excitation and inhibition
        for i, tpop in enumerate(['RecExc', 'RecInh', 'DsInh']): # both on excitation and inhibition
            ntwk.construct_feedforward_input(NTWK, tpop, 'AffExc',
                                             t_array, faff,
                                             verbose=True)

        ################################################################
        ## --------------- Initial Condition ------------------------ ##
        ################################################################
        ntwk.initialize_to_rest(NTWK)

        #####################
        ## ----- Run ----- ##
        #####################
        network_sim = ntwk.collect_and_run(NTWK, verbose=True)

        ntwk.write_as_hdf5(NTWK, filename='CellRep2019_data.h5')
        print('Results of the simulation are stored as:', 'CellRep2019_data.h5')
        print('--> Run \"python CellRep2019.py plot\" to plot the results')
