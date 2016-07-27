import numpy as np

def heaviside(x):
    return 0.5*(1+np.sign(x))

def smooth_heaviside(x):
    return 0.5*(1+erf(x))

def smooth_double_gaussian(t, t0, T1, T2, amplitude, smoothing=1e-2):
    return amplitude*(\
                      np.exp(-(t-t0)**2/2./T1**2)*smooth_heaviside(-(t-t0)/smoothing)+\
                      np.exp(-(t-t0)**2/2./T2**2)*smooth_heaviside((t-t0)/smoothing))

def double_gaussian(t, t0, T1, T2, amplitude):
    return amplitude*(\
                      np.exp(-(t-t0)**2/2./T1**2)*heaviside(-(t-t0))+\
                      np.exp(-(t-t0)**2/2./T2**2)*heaviside(t-t0))

