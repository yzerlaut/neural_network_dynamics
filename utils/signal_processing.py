import numpy.fft as fft
import numpy as np
from scipy import signal
from scipy import integrate
from scipy.ndimage.filters import gaussian_filter1d
from scipy.ndimage.filters import maximum_filter1d

def gaussian_smoothing(Signal, idt_sbsmpl=10):
    """Gaussian smoothing of the data"""
    return gaussian_filter1d(Signal, idt_sbsmpl)


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


def get_acf_time(Signal, dt,
                 min_time=1., max_time=100.,
                 acf=None,
                 procedure='integrate'):
    """
    returns the autocorrelation time of some fluctuations: Signal

    two methods: fitting of an exponential decay vs temporal integration

    """

    if acf is None:
        acf, shift = autocorrel(Signal, np.mean([max_time, min_time]), dt)

    if procedure=='fit':
        def func(X):
            return np.sum(np.abs(np.exp(-shift/X[0])-acf))
        res = minimize(func, [min_time],
                       bounds=[[min_time, max_time]], method='L-BFGS-B')
        return res.x[0]
    else:
        # we integrate
        shift = np.arange(len(acf))*dt
        return integrate.cumtrapz(acf, shift)[-1]


def crosscorrel(Signal1, Signal2, tmax, dt):
    """
    argument : Signal1 (np.array()), Signal2 (np.array())
    returns : np.array()
    take two Signals, and returns their crosscorrelation function 

    CONVENTION:
    --------------------------------------------------------------
    when the peak is in the past (negative t_shift)
    it means that Signal2 is delayed with respect to Signal 1
    --------------------------------------------------------------
    """
    if len(Signal1)!=len(Signal2):
        print('Need two arrays of the same size !!')
        
    steps = int(tmax/dt) # number of steps to sum on
    time_shift = dt*np.concatenate([-np.arange(1, steps)[::-1], np.arange(steps)])
    CCF = np.zeros(len(time_shift))
    for i in np.arange(steps):
        ccf = np.corrcoef(Signal1[:len(Signal1)-i], Signal2[i:])
        CCF[steps-1+i] = ccf[0,1]
    for i in np.arange(steps):
        ccf = np.corrcoef(Signal2[:len(Signal1)-i], Signal1[i:])
        CCF[steps-1-i] = ccf[0,1]
    return CCF, time_shift

def crosscorrel_norm(Signal1,Signal2):
    """
    computes the cross-correlation, and takes into account the boundary
    effects ! so normalizes by the weight of the bins !!
    the two array have t have the same size @
    
    argument : Signal1 (np.array()), Signal2 (np.array())
    returns : np.array()
    take two Signals, and returns their crosscorrelation function 
    """
    if Signal1.size!=Signal2.size:
        print("problem no equal size vectors !!")
    Signal1 = (Signal1-Signal1.mean())
    Signal2 = (Signal2-Signal2.mean())
    cr = signal.fftconvolve(Signal1,Signal2,"full")/Signal1.std()/Signal2.std()
    ww = np.linspace(Signal1.size,0,-1)
    bin_weight = np.concatenate((ww[::-1],ww[1:]))
    return cr/bin_weight

def butter_lowpass(Fcutoff, Facq, order=5):
    nyq = 0.5 * Facq
    normal_cutoff = Fcutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, Fcutoff, Facq, order=5):
    b, a = butter_lowpass(Fcutoff, Facq, order=order)
    y = signal.lfilter(b, a, data)
    return y

def butter_highpass(Fcutoff, Facq, order=5):
    nyq = 0.5 * Facq
    normal_cutoff = Fcutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, Fcutoff, Facq, order=5):
    b, a = butter_highpass(Fcutoff, Facq, order=order)
    y = signal.lfilter(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, Facq, order=5):
    nyq = 0.5 * Facq
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, Facq, order=5):
    b, a = butter_bandpass(lowcut, highcut, Facq, order=order)
    y = signal.lfilter(b, a, data)
    return y

def low_pass_by_convolve_with_exp(data, T, dt):
    """
    function to be worked out, normalization not ok
    """
    tt = np.arange(int(5.*T/dt))*dt
    tmid = tt[int(len(tt)/2.)]
    exp = np.array([np.exp(-(t2-tmid)/T) if t2>=tmid else 0 for t2 in tt])
    conv_number = signal.convolve(np.ones(len(data)),
                                  .5*(1+np.sign(tt-tmid)),
                                  mode='same')
    output = signal.convolve(data, exp,
                             mode='same')/conv_number
    return output

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the Signal.
    The Signal is prepared by introducing reflected copies of the Signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output Signal.
    
    input:
        x: the input Signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed Signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve
    scipy.Signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """ 
     
    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    
    y=np.convolve(w/w.sum(),s,mode='same')
    return y

def sliding_minimum(array, Window):
    return -maximum_filter1d(-array, size=Window)

def sliding_maximum(array, Window):
    return maximum_filter1d(array, size=Window)

def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))

def sliding_percentile(array, percentile, Window):

    x = np.zeros(len(array))
    y0 = strided_app(array, Window, 1)

    y = np.percentile(y0, percentile, axis=-1)
    
    x[:int(Window/2)] = y[0]
    x[-int(Window/2):] = y[-1]
    x[int(Window/2)-1:-int(Window/2)] = y
    
    return x


if __name__=='__main__':

    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    import utils.plot_tools as pt
     
    t = np.linspace(0, np.pi*2, int(1e3))
    Signal1 = np.sin(3*t)
    Signal2 = np.cos(3*t-np.pi/4)
    cr, t_shift = crosscorrel(Signal1, Signal2, np.pi/2, t[1]-t[0])
    _, ax = pt.figure(axes=(1,2), figsize=(1.5, 1.3))
    ax[0].plot(t, Signal1, 'k')
    ax[0].plot(t, Signal2)
    ax[0].plot(t, sliding_percentile(Signal2, 20, 200))
    ax[1].plot(t_shift, cr, '-')
    
    pt.plt.show()
