import os, sys
import numpy as np

import itertools, string, sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from datavyz.main import graph_env

from vision.earlyVis_model import params0
from vision.stimuli import setup_screen, screen_plot, screen_params0

def duration(distance, model):
    """
    from Baudot et al. (2013)

    The saccade amplitudes and intersaccadic intervals were chosen
    randomly from the distribution established for saccadic and head
    gaze movements in the freely behaving cat (Collewijn, 1977). An
    estimate of the duration of the saccade (D S ) was made by using
    the best linear fit between saccadic amplitude (A s ) and duration:
    D S = 1.9 × A S + 63
    (1)
    where D S is expressed in ms and A s in steradian degrees ( ◦ ) of
    visual angle.
    """
    return model['duration_distance_slope']*distance+\
        model['duration_distance_shift']


class virtual_eye_movement:

    def __init__(self,
                 params={'dt':5e-3, 'tstop':2,
                         'duration_distance_slope':1.9e-3, # degree/s
                         'duration_distance_shift':63e-3, # s
                         'rf_size':[0.2, 0.5],
                         'convolve_extent_factor':2.},
                 seed=1,
                 screen_params=None,
                 graph_env_key='visual_stim'):

        np.random.seed(seed)

        self.params = params
        if screen_params is not None:
            self.screen_params = screen_params
        else:
            self.screen_params = screen_params0 # from stimuli.py
            
        self.SCREEN = setup_screen(self.screen_params)
        
        self.ge = graph_env(graph_env_key)
        
        t = np.arange(int(params['tstop']/params['dt']))*params['dt'] # in seconds

        self.X = [self.SCREEN['width']/2.]
        self.Y = [self.SCREEN['height']/2.]
        self.events = [50e-3]
        self.x = 0*t+self.SCREEN['width']/2.
        self.y = 0*t+self.SCREEN['height']/2.

        while self.events[-1]<params['tstop']:
            self.X.append(np.random.uniform(params['rf_size'][1]*params['convolve_extent_factor'],\
                          self.SCREEN['width']-params['rf_size'][1]*params['convolve_extent_factor']))
            self.Y.append(np.random.uniform(params['rf_size'][1]*params['convolve_extent_factor'],\
                          self.SCREEN['height']-params['rf_size'][1]*params['convolve_extent_factor']))
            dd = duration(np.sqrt((self.X[-1]-self.X[-2])**2+\
                                  (self.Y[-1]-self.Y[-2])**2), params)
            self.events.append(self.events[-1]+dd)
            self.x[t>=self.events[-1]] = self.X[-1]
            self.y[t>=self.events[-1]] = self.Y[-1]

        

    def time_plot(self):

        t = np.arange(int(self.params['tstop']/self.params['dt']))*self.params['dt'] # in seconds
        self.ge.plot(t, Y=[self.x, self.y], LABELS=['X', 'Y'])

        # for i, ax in enumerate(np.array(AX).flatten()):
        #     ax.plot(X, Y, lw=1, color=ge.red)

        
vem = virtual_eye_movement()
vem.time_plot()
vem.ge.show()
