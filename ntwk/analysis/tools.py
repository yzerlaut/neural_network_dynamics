import numpy as np
from scipy.stats import skew
from scipy.optimize import minimize
from scipy.integrate import cumtrapz

def autocorrel(Signal, tmax, dt):
    """
    argument : Signal (np.array), tmax and dt (float)
    tmax, is the maximum length of the autocorrelation that we want to see
    returns : autocorrel (np.array), time_shift (np.array)
    take a Signal of time sampling dt, and returns its autocorrelation
     function between [0,tstop] (normalized) !!
    """
    steps = int(tmax/dt) # number of steps to sum on
    Signal2 = (Signal-Signal.mean())/Signal.std()
    cr = np.correlate(Signal2[steps:],Signal2)/steps
    time_shift = np.arange(len(cr))*dt
    return cr/cr.max(), time_shift

def get_acf_time(Vm, dt,
                 max_time_for_Tv=None,
                 min_time=1., max_time=1000.,
                 method='integrate'):
    """ 
    Autocorrelation time as the integral of the normalized autocorrelation function 
    """
    if max_time_for_Tv is None:
        max_time_for_Tv = len(Vm)*dt/10

    acf, shift = autocorrel(Vm, max_time_for_Tv, dt)
    if method=='integrate':
        return cumtrapz(acf, shift)[-1]
    elif method=='fit_exp':
        def func(X):
            return np.sum(np.abs(np.exp(-shift/X[0])-acf))
        res = minimize(func, [min_time],
                       bounds=[[min_time, max_time]], method='L-BFGS-B')
        return res.x[0]
    else:
        print('method: \"', method, '\" not implemented')
        return 0
    
if __name__=='__main__':
    Vm = np.random.randn(1000)
    print(get_acf_time(Vm, 0.1))
    print(get_acf_time(Vm, 0.1, method='fit_exp'))

