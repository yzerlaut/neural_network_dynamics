import numpy as np
import scipy.special as sp_spec
from itertools import product

## NORMALIZING COEFFICIENTS
# needs to be global here, because used both in the function
# and its derivatives
muV0, DmuV0 = -60e-3,10e-3
sV0, DsV0 = 4e-3, 6e-3
Tv0, DTv0 = 10e-3, 200e-3

def N_muV(muV):
    return (muV-muV0)/DmuV0
def N_sV(sV):
    return (sV-sV0)/DsV0
def N_Tv(Tv):
    return (Tv-Tv0)/DTv0
# identity when no normalization
def N_Id(x):
    return x 

NORM = [N_muV, N_sV, N_Id, N_Tv, N_Id] # array of normalizing functions

# effective threshold by inversion of the Firing rate data


def effective_Vthre(Fout, muV, sV, Tv):
    """
    inverting the basic firing approx 
    """
    Vthre_eff = muV+np.sqrt(2)*sV*sp_spec.erfcinv(Fout*2.*Tv)
    return Vthre_eff


def get_all_normalized_terms(muV, sV, gV, Tv, Proba, order=2):
    X = muV, sV, gV, Tv, Proba
    # we start with the 0 order term
    TERMS = [1.+0.*muV]
    # then first order
    if order>=1:
        for i in range(len(X)):
            TERMS.append(NORM[i](X[i]))
    # then first order
    if order>=2:
        for i,j in product(range(len(X)), range(len(X))):
            TERMS.append(NORM[i](X[i])*NORM[j](X[j]))
    #
    return TERMS

def firing_rate(muV, sV, gV, Tv, Proba, COEFFS, with_VTHRE_EFF=False):
    # terms 
    X = muV, sV, gV, Tv, Proba
    # we start with the 0 order term
    VTHRE_EFF = COEFFS[0]+0.*muV
    # then first order
    k=1
    if len(COEFFS)>0: # first order terms
        for i in range(len(X)):
            VTHRE_EFF += COEFFS[k]*NORM[i](X[i])
            k+=1
    # # then second order terms
    if len(COEFFS)>6:
        for i,j in product(range(len(X)), range(len(X))):
            VTHRE_EFF += COEFFS[k]*NORM[i](X[i])*NORM[j](X[j])
            k+=1
    if len(COEFFS)>31:
        print('third order not implemented')

    Fout = .5/Tv*sp_spec.erfc((VTHRE_EFF-muV)/np.sqrt(2)/sV)

    if with_VTHRE_EFF:
        return Fout, VTHRE_EFF
    else:
        return Fout

    
