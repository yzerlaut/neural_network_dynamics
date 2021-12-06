import os, sys
import numpy as np

import itertools, string, sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from datavyz.main import graph_env

from vision.stimuli import setup_screen, screen_params0

vem_params0 = {
    'saccade_duration_distance_slope':1.9e-3, # degree/s
    'saccade_duration_distance_shift':63e-3, # s
}

def saccade_duration(distance, model):
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
    return model['saccade_duration_distance_slope']*distance+\
        model['saccade_duration_distance_shift']


class virtual_eye_movement:

    def __init__(self,
                 eye_movement_key,
                 t,
                 boundary_extent_limit=1., # in degree
                 params=None,
                 seed=1,
                 screen_params=None):

        np.random.seed(seed)

        if params is not None:
            self.params = params
        else:
            self.params = vem_params0 # from stimuli.py
        
        if screen_params is not None:
            self.screen_params = screen_params
        else:
            self.screen_params = screen_params0 # from stimuli.py
            
        self.SCREEN = setup_screen(self.screen_params)
        self.boundary_extent_limit = boundary_extent_limit
        
        self.t = t

        if eye_movement_key=='saccadic':
            self.init_saccadic_eye_movement(seed+3)
        else:
            self.init_fixed_gaze_at_center()

            
    def init_fixed_gaze_at_center(self):
        self.x = 0*self.t+self.SCREEN['width']/2.
        self.y = 0*self.t+self.SCREEN['height']/2.
        self.events = []

    def init_saccadic_eye_movement(self, seed=1):
        """
        saccadic eve movement limited by the RF boundaries
        """
        self.X = [self.SCREEN['width']/2.]
        self.Y = [self.SCREEN['height']/2.]
        self.events = [50e-3]
        self.x = 0*self.t+self.SCREEN['width']/2.
        self.y = 0*self.t+self.SCREEN['height']/2.

        np.random.seed(seed)

        while self.events[-1]<self.t[-1]:
            self.X.append(np.random.uniform(self.boundary_extent_limit,\
                          self.SCREEN['width']-self.boundary_extent_limit))
            self.Y.append(np.random.uniform(self.boundary_extent_limit,\
                          self.SCREEN['height']-self.boundary_extent_limit))
            dd = saccade_duration(np.sqrt((self.X[-1]-self.X[-2])**2+\
                                  (self.Y[-1]-self.Y[-2])**2), self.params)
            self.events.append(self.events[-1]+dd)
            self.x[self.t>=self.events[-1]] = self.X[-1]
            self.y[self.t>=self.events[-1]] = self.Y[-1]

        

