import numpy as np
import scipy.special as sp_spec

# Proba_g_P = lambda mu_V, sigma_V, gamma_V, V_thre: -(gamma_V - gamma_V*(V_thre - mu_V)**2/sigma_V**2 + 3*np.sqrt(2)*np.sqrt(np.pi)*np.exp((V_thre - mu_V)**2/(2*sigma_V**2))*sp_spec.erf(np.sqrt(2)*(V_thre - mu_V)/(2*sigma_V)))*np.exp(-(V_thre - mu_V)**2/(2*sigma_V**2))/6 + np.sqrt(2)*np.sqrt(np.pi)/2

def Proba_g_P(mu_V, sigma_V, gamma_V, V_thre, XMAX=10):
    """ making a function to deal with too high values"""
    X_thre = (V_thre-mu_V)/sigma_V
    return X_thre**2*gamma_V*np.exp(-X_thre**2/2)/6 - gamma_V*np.exp(-X_thre**2/2)/6 - np.sqrt(2)*np.sqrt(np.pi)*sp_spec.erf(np.sqrt(2)*X_thre/2)/2 + np.sqrt(2)*np.sqrt(np.pi)/2
    
