import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from theory.psp_integrals import F_iPSP, F_iiPSP, F_iiiPSP, F_numTv, F_denomTv

def getting_statistical_properties(params, SYN_POPS, RATES, already_SI=False):
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
        else:
            SYN_PARAMS.append({'E_j': 1e-3*syn['Erev'], 'C_m':1e-12*params['Cm'],
                               'Q_j':syn['Q']*1e-9, 'tau_j':1e-3*syn['Tsyn']})
        RATES2.append(RATES['F_'+syn['name']]*syn['N']*syn['pconn'])

    # A zero array to handle both float and array cases (for addition/multiplication)
    Zero = 0.*RATES2[0] + 0.*SYN_POPS[0]['Q']
    
    # starting by the mean-dependent quantities: muV and Tm
    if already_SI:
        Gtot, muV = params['Gl']+Zero, params['Gl']*params['El']+Zero
    else:
        Gtot, muV = params['Gl']*1e-9+Zero, params['Gl']*params['El']*1e-12+Zero
        
    for i, syn in enumerate(SYN_PARAMS):
        Gsyn = RATES2[i]*syn['tau_j']*syn['Q_j']
        Gtot += Gsyn
        muV += Gsyn*syn['E_j']
    muV /= Gtot

    # from this we can get the mean membrane time constant
    if already_SI:
        Tm = params['Cm']/Gtot # 'Cm' from F to pF
    else:
        Tm = params['Cm']*1e-12/Gtot # 'Cm' from F to pF

    # we now have the mean properties, we can get the higher moments
    sV, gV, kV, nTv, dTv = 0, 0, 0, 0, 0
    for i, syn in enumerate(SYN_PARAMS):
        syn['mu_V'], syn['tau_m'] = muV, Tm
        sV += RATES2[i]*F_iiPSP(**syn)
        gV += RATES2[i]*F_iiiPSP(**syn)
        # kV += RATES2[i]*F_iiiiPSP(**syn)
        nTv += RATES2[i]*F_numTv(**syn)
        dTv += RATES2[i]*F_denomTv(**syn)
    sV = np.sqrt(sV)
    gV = gV/sV**3
    # kV = kV/sV**4
    Tv = 1./2.*(nTv/dTv)**(-1)
    # return muV, sV, gV, kV, Tv # with the kurtosis
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
