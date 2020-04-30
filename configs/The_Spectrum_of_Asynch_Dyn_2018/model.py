import numpy as np

Model = {
    ## ---------------------------------------------------------------------------------
    ### Initialisation by default parameters
    ## UNIT SYSTEM is : ms, mV, pF, nS, pA, Hz (arbitrary and unconsistent, so see code)
    ## ---------------------------------------------------------------------------------
    # numbers of neurons in population
    'REC_POPS':['RecExc', 'RecInh', 'DsInh'],
    'AFF_POPS':['AffExc'],
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
    'DsInh_a':0., 'DsInh_b': 0., 'DsInh_tauw':1e9,
    # COEFFS for MF
    'COEFFS_RecExc' : np.load('COEFFS_RecExc.npy'),
    'COEFFS_RecInh' : np.load('COEFFS_RecInh.npy'),
    'COEFFS_DsInh' : np.load('COEFFS_RecExc.npy'),
    #
}
Model['AffExc_IncreasingStep_onset']= 1000 # 200ms delay for onset
Model['AffExc_IncreasingStep_baseline']= 0.
Model['AffExc_IncreasingStep_length']= 1000
Model['AffExc_IncreasingStep_size']= 4.
Model['AffExc_IncreasingStep_smoothing']= 100

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

        # ntwk.quick_ntwk_sim(Model)
        ntwk.quick_MF_sim(Model)
