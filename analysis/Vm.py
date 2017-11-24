import numpy as np
import matplotlib.pylab as plt
from itertools import combinations # for cross correlations
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from scipy.stats import skew
from data_analysis.processing.signanalysis import gaussian_smoothing,\
    autocorrel, butter_highpass_filter
from scipy.integrate import cumtrapz

def get_acf_time(Vm, dt, max_time_for_Tv=100.):
    """ 
    Autocorrelation time as the integral of the normalized autocorrelation function 
    """
    acf, shift = autocorrel(Vm, max_time_for_Tv, dt)
    return cumtrapz(acf, shift)[-1]

def fluctuations_properties(Vm, dt, max_time_for_Tv=100.):
    """ return the fluctuations properties of a given Vm samples, see Zerlaut et al. 2018"""
    return np.mean(Vm), np.std(Vm), skew(np.array(Vm)),\
        get_acf_time(Vm, dt, max_time_for_Tv=max_time_for_Tv)

if __name__=='__main__':
    Vm = np.random.randn(10000)
    print(fluctuations_properties(Vm, 1, max_time_for_Tv=10))
