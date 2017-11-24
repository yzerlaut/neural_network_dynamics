import numpy as np
import matplotlib.pylab as plt
from itertools import combinations # for cross correlations
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from scipy.stats import skew
from data_analysis.processing.signanalysis import gaussian_smoothing,\
    autocorrel, butter_highpass_filter
from scipy.integrate import cumtrapz
from scipy.optimize import minimize
from graphs.my_graph import *

def get_acf_time(Vm, dt,
                 max_time_for_Tv=100.,
                 min_time=1., max_time=1000.,
                 method='integrate'):
    """ 
    Autocorrelation time as the integral of the normalized autocorrelation function 
    """
    acf, shift = autocorrel(Vm, max_time_for_Tv, dt)
    if method is 'integrate':
        return cumtrapz(acf, shift)[-1]
    elif method is 'fit_exp':
        def func(X):
            return np.sum(np.abs(np.exp(-shift/X[0])-acf))
        res = minimize(func, [min_time],
                       bounds=[[min_time, max_time]], method='L-BFGS-B')
        return res.x[0]

def fluctuations_properties(Vm, dt,
                            max_time_for_Tv=100.,
                            min_time=1., max_time=1000.,
                            method='integrate',
                            with_plot=False):
    """ return the fluctuations properties of a given Vm samples, see Zerlaut et al. 2018"""
    if with_plot:
        fig, AX = plt.subplots(1, 2, figsize=(6,3))
        plt.subplots_adjust(wspace=.7, bottom=0.3)
        AX[0].hist(Vm, bins=100)
        acf, shift = autocorrel(Vm, max_time_for_Tv, dt)
        AX[1].plot(shift, acf)
        for ax in AX: set_plot(ax)
        show()
        
    return np.mean(Vm), np.std(Vm), skew(Vm),\
        get_acf_time(Vm, dt, max_time_for_Tv=max_time_for_Tv,
                     method=method,
                     min_time=min_time, max_time=max_time)

if __name__=='__main__':
    Vm = np.load('temp.npy').flatten()
    print(fluctuations_properties(Vm, 0.1, max_time_for_Tv=100, method='fit_exp', with_plot=True))
