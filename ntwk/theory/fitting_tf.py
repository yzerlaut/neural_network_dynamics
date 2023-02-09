import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from theory.Vm_statistics import getting_statistical_properties
from theory.probability import Proba_g_P
from theory.spiking_function import effective_Vthre, get_all_normalized_terms
from theory.tf import build_up_afferent_synaptic_input, built_up_neuron_params
from sklearn import linear_model
import numpy as np

def fit_data(data, order=2, Fout_high=50., fit_filename=None):
    
    mFout, sFout = data['Fout_mean'], data['Fout_std']
    Model = data['Model']

    neuron_params = built_up_neuron_params(Model, Model['NRN_KEY'])
    SYN_POPS = build_up_afferent_synaptic_input(Model, Model['POP_STIM'])
    
    RATES={}
    for syn in SYN_POPS:
        RATES['F_'+syn['name']] = data['F_'+syn['name']]
        
    ### OUTPUT OF ANALYTICAL CALCULUS IN SI UNITS !! -> from here SI, be careful...
    muV, sV, gV, Tv = getting_statistical_properties(neuron_params,
                                                     SYN_POPS, RATES,
                                                     already_SI=False)
    Proba = Proba_g_P(muV, sV, gV, 1e-3*neuron_params['Vthre'])

    # # only strictly positive firing rates taken into account
    cond = (mFout>0) & (mFout<Fout_high)
    
    Fout, muV, sV, gV, Tv, Proba = mFout[cond], muV[cond], sV[cond], gV[cond], Tv[cond], Proba[cond]

    # # computing effective threshold
    Vthre_eff = effective_Vthre(Fout, muV, sV, Tv)

    finite_cond = ~np.isnan(Vthre_eff)

    TERMS = get_all_normalized_terms(muV[finite_cond], sV[finite_cond],
                                     gV[finite_cond], Tv[finite_cond],
                                     Proba[finite_cond],
                                     order=order)

    X = np.array(TERMS).T

    # first the linear fitting
    reg = linear_model.LinearRegression(fit_intercept=False, normalize=False)
    reg.fit(X, Vthre_eff[finite_cond]) # FITTING
    COEFFS = reg.coef_

    reg = linear_model.Ridge(alpha=.9, fit_intercept=False, normalize=False)
    reg.fit(X, Vthre_eff[finite_cond]) # FITTING
    COEFFS = reg.coef_

    if fit_filename is not None:
        np.save(fit_filename, COEFFS)
        
    return COEFFS
    

def set_equally_sampled_data_on_firing_range(data,
                                             firing_rate_range = np.logspace(-2,2,4)):

    
    new_data = {'Model':data['Model']}
    keys = data.keys()
    keys.remove('Model')
    
    Nsamples = []
    for y1, y2 in zip(firing_rate_range[:-1], firing_rate_range[1:]):
        cond = (data['Fout_mean']>=y1) & (data['Fout_mean']<y2)
        Nsamples.append(len(data['Fout_mean'][cond]))

    Nmin = min(Nsamples)

    # for key, val in data.items():
    #     if key!='Model':
    #         new_data.
    # for i, (y1, y2) in enumerate(zip(firing_rate_range[:-1], firing_rate_range[1:])):
    #     cond = (data['Fout_mean']>=y1) & (data['Fout_mean']<y2)
    #     choosen_samples = np.random.choice(np.arange(Nsamples), Nmin, replace=False)
    # ['F_RecInh', 'F_RecExc', 'F_AffExc', 'Model', 'Fout_std', 'Fout_mean', 'F_DsInh']

    
    return new_data


