import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from theory.psp_integrals import F_iPSP, F_iiPSP, F_iiiPSP, F_numTv, F_denomTv

def getting_statistical_properties(params,
                                   SYN_POPS, RATES,
                                   already_SI=False, with_Isyn=False, with_current_based=False):
    """ 
    We first translate those parameters into SI units for a safe calculus
    then we apply the results of the Shotnoise analysis (see above)
    and we return the three moments
    """
    SYN_PARAMS, RATES2 = [], []

    for i, syn in enumerate(SYN_POPS):
        if already_SI:
            SYN_PARAMS.append({'E_j': syn['Erev'], 'C_m':params['Cm'],
                               'Q_j':syn['Q'], 'tau_j':syn['Tsyn']})
            if 'V0' in syn:
                SYN_PARAMS[-1]['V0'] = syn['V0']
            else:
                SYN_PARAMS[-1]['V0'] = 0
        else:
            SYN_PARAMS.append({'E_j': 1e-3*syn['Erev'], 'C_m':1e-12*params['Cm'],
                               'Q_j':syn['Q']*1e-9, 'tau_j':1e-3*syn['Tsyn']})
            if 'V0' in syn:
                SYN_PARAMS[-1]['V0'] = 1e-3*syn['V0']
            else:
                SYN_PARAMS[-1]['V0'] = 0
                
        if 'alpha' in syn:
            SYN_PARAMS[-1]['a_j'] = syn['alpha']
        else:
            SYN_PARAMS[-1]['a_j'] = 1 # pure conductance based by default
        RATES2.append(RATES['F_'+syn['name']]*syn['N']*syn['pconn'])

    # A zero array to handle both float and array cases (for addition/multiplication)
    Zero = 0.*RATES2[0] + 0.*SYN_POPS[0]['Q']
    
    # starting by the mean-dependent quantities: muV and Tm
    if already_SI:
        Gtot, muV = params['Gl']+Zero, params['Gl']*params['El']+Zero
    else:
        Gtot, muV = params['Gl']*1e-9+Zero, params['Gl']*params['El']*1e-12+Zero

    for i, syn in enumerate(SYN_PARAMS):
        Gsyn = RATES2[i]*syn['tau_j']*syn['Q_j']*syn['a_j']
        Gtot += Gsyn
        Isyn = RATES2[i]*syn['tau_j']*syn['Q_j']*(1-syn['a_j'])*(syn['E_j']-syn['V0'])
        muV += Gsyn*syn['E_j']+Isyn
    muV /= Gtot

    # from this we can get the mean membrane time constant
    if already_SI:
        Tm = params['Cm']/Gtot # 'Cm' from F to pF
        Tm0 = params['Cm']/params['Gl']
    else:
        Tm = params['Cm']*1e-12/Gtot # 'Cm' from F to pF
        Tm0 = 1e-3*params['Cm']/params['Gl']

    # we now have the mean properties, we can get the higher moments
    sV, gV, kV, nTv, dTv = 0, 0, 0, 0, 0
    for i, syn in enumerate(SYN_PARAMS):
        syn['mu_V'], syn['tau_m'] = muV, Tm
        sV += RATES2[i]*F_iiPSP(**syn)
        gV += RATES2[i]*F_iiiPSP(**syn)
        # kV += RATES2[i]*F_iiiiPSP(**syn)
        nTv += RATES2[i]*F_numTv(**syn)
        dTv += RATES2[i]*F_denomTv(**syn)
    sV[sV<1e-6] = 1e-6 # thresholded to 0.001 mV
    sV = np.sqrt(sV) 
    gV = gV/sV**3
    # kV = kV/sV**4
    Tv = Tm # by default
    Tv[dTv>0] = 1./2.*(nTv/dTv)**(-1)

    if with_Isyn:
        # in case we also want synaptic currents
        Isyn = {}
        for i, syn in enumerate(SYN_PARAMS):
            Isyn[SYN_POPS[i]['name']] = RATES2[i]*syn['tau_j']*syn['Q_j']*\
                                        (syn['a_j']*(syn['E_j']-muV)+(1-syn['a_j'])*(syn['E_j']-syn['V0']))
        if with_current_based:
            sV0 = 0
            for i, syn in enumerate(SYN_PARAMS):
                syn['mu_V'], syn['tau_m'] = muV, Tm0
                sV0 += RATES2[i]*F_iiPSP(**syn)
            Isyn['sV0'] = np.sqrt(sV0)
        return muV, sV, gV, Tv, Isyn
    else:
        return muV, sV, gV, Tv

def distribution(vv, muV, sV, gV, kV=0, with_edgeworth=True):
    """
    """
    D = 1./np.sqrt(2.*np.pi)/sV*np.exp(-(vv-muV)**2/2./sV**2)
    if with_edgeworth:
        kurt_term = kV/24.*( ((vv-muV)/sV)**4 - 6.*((vv-muV)/sV)**2 ) # + 3
        skew_term = gV/6.*( ((vv-muV)/sV)**3 - 3.*(vv-muV)/sV )
        D *= 1+skew_term+kurt_term
    return D
