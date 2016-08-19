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

def ramp_rise_then_constant(t, t1, t2, amp1, amp2):
    return 0*t+amp1+heaviside(t-t1)*heaviside(t2-t)*(t-t1)/(t2-t1)*(amp2-amp1)+\
        heaviside(t-t2)*(amp2-amp1)
    
if __name__=='__main__':
    import matplotlib.pylab as plt
    t = np.linspace(0, 200, 1e3)
    x = ramp_rise_then_constant(t, 50, 100, 50, 200)
    x = double_gaussian(t_array, 60., 30., 20., 10.)
    plt.plot(t, x)
    plt.show()
    
