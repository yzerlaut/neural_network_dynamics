
Model = {
    ## ---------------------------------------------------------------------------------
    ### Initialisation by default parameters
    ## UNIT SYSTEM is : ms, mV, pF, nS, pA, Hz (arbitrary and unconsistent, so see code)
    ## ---------------------------------------------------------------------------------
    # numbers of neurons in population
    'N_RecExc':4000, 'N_RecInh':1000, 'N_AffExc':100, 'N_DsInh':500,
    # synaptic weights
    'Q_RecExc_RecExc':2., 'Q_RecExc_RecInh':2., 
    'Q_RecInh_RecExc':10., 'Q_RecInh_RecInh':10., 
    'Q_AffExc_RecExc':4., 'Q_AffExc_RecInh':4., 
    'Q_AffExc_DsInh':4.,
    'Q_DsInh_RecInh':10., 
    # synaptic time constants
    'Tse':5., 'Tsi':5.,
    # synaptic reversal potentials
    'Ee':0., 'Ei': -80.,
    # connectivity parameters
    'p_RecExc_RecExc':0.05, 'p_RecExc_RecInh':0.05, 
    'p_RecInh_RecExc':0.05, 'p_RecInh_RecInh':0.05, 
    'p_AffExc_RecExc':0.1, 'p_AffExc_RecInh':0.1, 
    'p_AffExc_DsInh':0.,
    'p_DsInh_RecInh':0.05, 
    # afferent stimulation
    'F_AffExc':10., 'F_DsInh':0.,
    # recurrent activity (for single cell simulation only)
    'F_RecExc':1., 'F_RecInh':1.,
    # simulation parameters
    'dt':0.1, 'tstop': 100., 'SEED':3, # low by default, see later
    ## ---------------------------------------------------------------------------------
    # === cellular properties (based on AdExp), population by population ===
    # --> Excitatory population (RecExc, recurrent excitation)
    'RecExc_Gl':10., 'RecExc_Cm':200.,'RecExc_Trefrac':5.,
    'RecExc_El':-70., 'RecExc_Vthre':-50., 'RecExc_Vreset':-70., 'RecExc_delta_v':0.,
    'RecExc_a':0., 'RecExc_b': 0., 'RecExc_tauw':1e9,
    # --> Inhibitory population (RecInh, recurrent inhibition)
    'RecInh_Gl':10., 'RecInh_Cm':200.,'RecInh_Trefrac':5.,
    'RecInh_El':-70., 'RecInh_Vthre':-53., 'RecInh_Vreset':-70., 'RecInh_delta_v':0.,
    'RecInh_a':0., 'RecInh_b': 0., 'RecInh_tauw':1e9,
    # --> Disinhibitory population (DsInh, disinhibition)
    'DsInh_Gl':10., 'DsInh_Cm':200.,'DsInh_Trefrac':5.,
    'DsInh_El':-70., 'DsInh_Vthre':-50., 'DsInh_Vreset':-70., 'DsInh_delta_v':0.,
    'DsInh_a':0., 'DsInh_b': 0., 'DsInh_tauw':1e9
}

# now let's say that we will modify or add a parameter in 'Model', we will execute scripts with the new statement:
def pass_arguments_of_new_model(Model):
    S = ''
    for key, val in Model.items():
        S += ' --'+key+' '+str(val)
    return S

import argparse
parser=argparse.ArgumentParser(description='Model parameters',formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--Q_AffExc_RecExc", type=float, default=4.0) 
parser.add_argument("--p_AffExc_RecExc", type=float, default=0.1) 
parser.add_argument("--RecExc_Vreset", type=float, default=-70.0) 
parser.add_argument("--RecInh_Vreset", type=float, default=-70.0) 
parser.add_argument("--p_AffExc_RecInh", type=float, default=0.1) 
parser.add_argument("--DsInh_Vreset", type=float, default=-70.0) 
parser.add_argument("--N_DsInh", type=int, default=500) 
parser.add_argument("--DsInh_b", type=float, default=0.0) 
parser.add_argument("--N_AffExc", type=int, default=100) 
parser.add_argument("--RecInh_El", type=float, default=-70.0) 
parser.add_argument("--DsInh_tauw", type=float, default=1000000000.0) 
parser.add_argument("--RecExc_tauw", type=float, default=1000000000.0) 
parser.add_argument("--Q_RecInh_RecExc", type=float, default=10.0) 
parser.add_argument("--F_DsInh", type=float, default=0.0) 
parser.add_argument("--RecInh_Cm", type=float, default=200.0) 
parser.add_argument("--DsInh_Vthre", type=float, default=-50.0) 
parser.add_argument("--RecInh_a", type=float, default=0.0) 
parser.add_argument("--F_RecExc", type=float, default=1.0) 
parser.add_argument("--RecInh_b", type=float, default=0.0) 
parser.add_argument("--RecExc_El", type=float, default=-70.0) 
parser.add_argument("--F_RecInh", type=float, default=1.0) 
parser.add_argument("--RecExc_delta_v", type=float, default=0.0) 
parser.add_argument("--RecExc_Trefrac", type=float, default=5.0) 
parser.add_argument("--p_RecInh_RecExc", type=float, default=0.05) 
parser.add_argument("--SEED", type=int, default=3) 
parser.add_argument("--Ei", type=float, default=-80.0) 
parser.add_argument("--DsInh_Gl", type=float, default=10.0) 
parser.add_argument("--N_RecExc", type=int, default=4000) 
parser.add_argument("--RecExc_a", type=float, default=0.0) 
parser.add_argument("--p_RecExc_RecInh", type=float, default=0.05) 
parser.add_argument("--RecExc_Vthre", type=float, default=-50.0) 
parser.add_argument("--RecInh_Gl", type=float, default=10.0) 
parser.add_argument("--F_AffExc", type=float, default=10.0) 
parser.add_argument("--DsInh_delta_v", type=float, default=0.0) 
parser.add_argument("--DsInh_a", type=float, default=0.0) 
parser.add_argument("--Q_RecInh_RecInh", type=float, default=10.0) 
parser.add_argument("--p_DsInh_RecInh", type=float, default=0.05) 
parser.add_argument("--RecInh_tauw", type=float, default=1000000000.0) 
parser.add_argument("--RecInh_Trefrac", type=float, default=5.0) 
parser.add_argument("--RecExc_Gl", type=float, default=10.0) 
parser.add_argument("--Q_DsInh_RecInh", type=float, default=10.0) 
parser.add_argument("--Q_RecExc_RecInh", type=float, default=2.0) 
parser.add_argument("--Tse", type=float, default=5.0) 
parser.add_argument("--RecInh_delta_v", type=float, default=0.0) 
parser.add_argument("--DsInh_Cm", type=float, default=200.0) 
parser.add_argument("--N_RecInh", type=int, default=1000) 
parser.add_argument("--p_RecInh_RecInh", type=float, default=0.05) 
parser.add_argument("--p_RecExc_RecExc", type=float, default=0.05) 
parser.add_argument("--Tsi", type=float, default=5.0) 
parser.add_argument("--Q_AffExc_DsInh", type=float, default=4.0) 
parser.add_argument("--DsInh_Trefrac", type=float, default=5.0) 
parser.add_argument("--Ee", type=float, default=0.0) 
parser.add_argument("--RecExc_Cm", type=float, default=200.0) 
parser.add_argument("--RecExc_b", type=float, default=0.0) 
parser.add_argument("--Q_AffExc_RecInh", type=float, default=4.0) 
parser.add_argument("--p_AffExc_DsInh", type=float, default=0.0) 
parser.add_argument("--tstop", type=float, default=100.0) 
parser.add_argument("--Q_RecExc_RecExc", type=float, default=2.0) 
parser.add_argument("--dt", type=float, default=0.1) 
parser.add_argument("--DsInh_El", type=float, default=-70.0) 
parser.add_argument("--RecInh_Vthre", type=float, default=-53.0) 
