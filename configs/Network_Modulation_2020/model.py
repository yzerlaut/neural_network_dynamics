################################################################
## ------ Construct populations with their equations -------- ##
## ------------- with recurrent connections ----------------- ##
################################################################


# adding the same LIF props to all recurrent pops
LIF_props = {'Gl':10., 'Cm':200.,'Trefrac':5.,
             'El':-70, 'Vthre':-50., 'Vreset':-70., 'deltaV':0.,
             'a':0., 'b': 0., 'tauw':1e9}

Model = {
    ## -----------------------------------------
    ### Model properties
    ## -----------------------------------------
    ## UNIT SYSTEM is : ms, mV, pF, nS, pA, Hz (arbitrary and unconsistent, so see code)
    ## ------------------------------------------
    ##
    ## POPULATIONS
    'REC_POPS':['pyrExc', 'oscillExc', 'recInh', 'dsInh'],
    'AFF_POPS':['AffExc', 'NoiseExc'],
    # numbers of neurons in population
    'N_pyrExc':4000, 'N_recInh':1000, 'N_dsInh':500, 'N_oscillExc':100,
    'N_AffExc':100, 'N_NoiseExc':200,
    # synaptic time constants
    'Tse':5., 'Tsi':5.,
    # synaptic reversal potentials
    'Ee':0., 'Ei': -80.,
    # afferent stimulation
    'F_AffExc':10., 'F_NoiseExc':3.,
    # simulation parameters
    'dt':0.1, 'tstop': 6000., 'SEED':3, # low by default, see later
}

for pop in Model['REC_POPS']:
    for key, val in LIF_props.items():
        Model['%s_%s' % (pop, key)] = val
# adding the oscillatory feature to the oscillExc pop
Model['oscillExc_Ioscill_freq']=3.
Model['oscillExc_Ioscill_amp']= 10.*15.
Model['oscillExc_Ioscill_onset']= 0. 

Model['AffExc_IncreasingStep_onset']= 1000 # 200ms delay for onset
Model['AffExc_IncreasingStep_baseline']= 0.
Model['AffExc_IncreasingStep_length']= 1000
Model['AffExc_IncreasingStep_size']= 4.
Model['AffExc_IncreasingStep_smoothing']= 100

# === adding synaptic weights ===
Qe, Qi = 2., 10 # nS
# loop oover the two population types
for aff in ['pyrExc', 'oscillExc', 'NoiseExc', 'AffExc']:
    for target in Model['REC_POPS']:
        Model['Q_%s_%s' % (aff, target)] = Qe
for aff in ['recInh', 'dsInh']:
    for target in Model['REC_POPS']:
        Model['Q_%s_%s' % (aff, target)] = Qi

# === initializing connectivity === #         
for aff in Model['REC_POPS']+Model['AFF_POPS']:
    for target in Model['REC_POPS']:
        Model['p_%s_%s' % (aff, target)] = 0.01 # minimal connectivity by default

# -------------------------------
# --- connectivity parameters ---
# -------------------------------
# # ==> pyrExc
# Model['p_pyrExc_pyrExc'] = 0.01
# Model['p_pyrExc_oscillExc'] = 0.01
# Model['p_pyrExc_recInh'] = 0.02
# # ==> oscillExc
# Model['p_oscillExc_oscillExc'] = 0.01
# Model['p_oscillExc_pyrExc'] = 0.05
# # Model['p_oscillExc_recInh'] = 0.02
# # ==> recInh
# Model['p_recInh_recInh'] = 0.05
# Model['p_recInh_pyrExc'] = 0.05
# Model['p_recInh_oscillExc'] = 0.1
# # ==> dsInh
# Model['p_dsInh_recInh'] = 0.1
# # ==> AffExc
# Model['p_AffExc_dsInh'] = 0.2
# Model['p_AffExc_recInh'] = 0.2
# Model['p_AffExc_pyrExc'] = 0.05
# # ==> NoiseExc
# Model['p_NoiseExc_recInh'] = 0.1
# Model['p_NoiseExc_pyrExc'] = 0.02
# Model['p_NoiseExc_dsInh'] = 0.02
# Model['p_NoiseExc_oscillExc'] = 0.2

Model['p_pyrExc_oscillExc'] = 0.062
Model['p_pyrExc_recInh'] = 0.044
Model['p_pyrExc_dsInh'] = 0.012
Model['p_oscillExc_pyrExc'] = 0.302
Model['p_oscillExc_oscillExc'] = 0.066
Model['p_oscillExc_recInh'] = 0.175
Model['p_oscillExc_dsInh'] = 0.423
Model['p_recInh_pyrExc'] = 0.084
Model['p_recInh_oscillExc'] = 0.096
Model['p_recInh_recInh'] = 0.061
Model['p_recInh_dsInh'] = 0.067
Model['p_dsInh_pyrExc'] = 0.030
Model['p_dsInh_oscillExc'] = 0.051
Model['p_dsInh_recInh'] = 0.045
Model['p_dsInh_dsInh'] = 0.039
Model['p_AffExc_pyrExc'] = 0.163
Model['p_AffExc_oscillExc'] = 0.058
Model['p_AffExc_recInh'] = 0.149
Model['p_AffExc_dsInh'] = 0.212
Model['p_NoiseExc_pyrExc'] = 0.052
Model['p_NoiseExc_oscillExc'] = 0.082
Model['p_NoiseExc_recInh'] = 0.149
Model['p_NoiseExc_dsInh'] = 0.114

if __name__=='__main__':

    print(Model)        
