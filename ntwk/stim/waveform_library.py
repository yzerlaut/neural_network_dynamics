import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from scipy.special import erf

def heaviside(x):
    return 0.5*(1+np.sign(x))

def smooth_heaviside(x):
    return 0.5*(1+erf(x))

def gaussian_smoothing(Signal, idt_sbsmpl=10):
    """Gaussian smoothing of the data"""
    return gaussian_filter1d(Signal, idt_sbsmpl)

def smooth_double_gaussian(t, t0, T1, T2, amplitude, smoothing=1e-2):
    return amplitude*(\
                      np.exp(-(t-t0)**2/2./T1**2)*smooth_heaviside(-(t-t0)/smoothing)+\
                      np.exp(-(t-t0)**2/2./T2**2)*smooth_heaviside((t-t0)/smoothing))

def double_gaussian(t, t0, T1, T2, amplitude):
    return amplitude*(\
                      np.exp(-(t-t0)**2/2./T1**2)*heaviside(-(t-t0))+\
                      np.exp(-(t-t0)**2/2./T2**2)*heaviside(t-t0))

def gaussian(t, t0, sT, amplitude):
    gauss = np.exp(-(t-t0)**2/2./sT**2)
    return amplitude*gauss/gauss.max()

def ramp_rise_then_constant(t, t1, t2, amp1, amp2, translate_to_SI=False):
    if translate_to_SI:
        Tfactor=1e-3
    else:
        Tfactor=1
    return 0*t+amp1+heaviside(t-t1)*heaviside(t2-t)*(t-t1)/(t2-t1)*(amp2-amp1)+\
        heaviside(t-t2)*(amp2-amp1)

def delayed_oscillation(t, onset, amp, freq, already_SI=False):
    if not already_SI:
        Tfactor=1e-3
    else:
        Tfactor=1
    signal = 0*t
    cond = t>onset
    signal[cond] = amp*(1-np.cos(2.*np.pi*freq*Tfactor*(t[cond]-t[cond][0])))/2.
    return signal

def increasing_step_function(t, baseline, onset, size, length, 
                             smoothing=0):
    signal = 0*t+baseline
    i, t0 = 1, onset
    while t0<t[-1]:
        signal[t>t0] =baseline+i*size
        t0+=length
        i+=1

    if smoothing>0:
        return gaussian_smoothing(signal, int(smoothing/(t[1]-t[0])))
    else:
        return signal

def varying_levels_function(t, levels, onsets, 
                            smoothing=100):

    signal = 0*t

    for onset, level in zip(onsets, levels):
        signal[t>onset] = level

    return gaussian_smoothing(signal, int(smoothing/(t[1]-t[0])))

############################################################
# -------  Using the network "pop" key framework  ----------
############################################################

def Intrinsic_Oscill(t, pop, Model, translate_to_SI=False):
    if translate_to_SI:
        Tfactor=1e-3
    else:
        Tfactor=1
    return delayed_oscillation(t,
                               Tfactor*Model['%s_Ioscill_onset'%pop],
                               Model['%s_Ioscill_amp'%pop],
                               Model['%s_Ioscill_freq'%pop], already_SI=translate_to_SI)


def IncreasingSteps(t, pop, Model, translate_to_SI=False):
    if translate_to_SI:
        Tfactor=1e-3
    else:
        Tfactor=1
    return increasing_step_function(t,
                                    Model['%s_IncreasingStep_baseline'%pop],
                                    Tfactor*Model['%s_IncreasingStep_onset'%pop],
                                    Model['%s_IncreasingStep_size'%pop],
                                    Tfactor*Model['%s_IncreasingStep_length'%pop],
                                    Tfactor*Model['%s_IncreasingStep_smoothing'%pop])




if __name__=='__main__':
    import matplotlib.pylab as plt
    t = np.linspace(0, 200, 1e3)
    x = ramp_rise_then_constant(t, 50, 100, 50, 200)
    x = double_gaussian(t, 60., 30., 20., 10.)
    plt.plot(t, x)
    plt.show()

