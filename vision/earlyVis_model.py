import numpy as np

import itertools, string, sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from datavyz.main import graph_env
from vision.gabor_filters import gabor
from vision.stimuli import setup_screen, screen_plot, screen_params0, stim_params0, visual_stimulus
from vision.virtual_eye_movement import virtual_eye_movement, vem_params0

params0 = {
    #
    'Ncells':2,
    # receptive fields
    'rf_fraction':.4, # fraction of visual space covered by the cells, fraction of the screen 
    'rf_size':[0.2, 0.5], # degrees
    'rf_freq':[0.6, 1.], # cycle per degrees
    'rf_beta':[1., 2.],
    'rf_theta':[0., np.pi/2.],
    'rf_psi':[0., np.pi],
    'convolve_extent_factor':2., # limit the convolution to thisfactor*rf-width to make comput faster
    # temporal filtering
    'tau_adapt':500e-3,
    'tau_delay':30e-3,
    'fraction_adapt':0.2,
    # non-linear amplification
    'NL_threshold':0.1,
    'NL_slope_Hz_per_Null':10.,
    # # virtual eye movement
    'duration_distance_slope':1.9e-3, # degree/s
    'duration_distance_shift':63e-3, # s
    # simulation params
    'dt':10e-3,
    'tstop':1,
}

full_params0 = {**params0, **screen_params0, **stim_params0, **vem_params0}

class earlyVis_model:
    """
    A model of the early visual system
    from luminance in the visual space to spikes in layer IV pyramidal cells in V1
    Steps:
    - linear sptial filtering (Gabor filters)
    - temporal filtering (delay and adaptation)
    - non-linear transformation (to get firing rates)
    - Poisson process transofrmation to get spikes
    """
    
    def __init__(self,
                 params=None,
                 graph_env_key='visual_stim'):
        
        if params is not None:
            self.params = params
        else:
            self.params = full_params0 # above params by default

        self.SCREEN = setup_screen(self.params)
        
        self.setup_RF_props()

        self.visual_stim = None
        self.dt_screen = 1./self.params['screen_refresh_rate']
        self.t_screen = np.arange(int(self.params['tstop']/self.dt_screen))*self.dt_screen
        
        self.eye_movement = None
        
        self.ge = graph_env(graph_env_key)
        
        self.t = np.arange(int(self.params['tstop']/self.params['dt']))*self.params['dt']

        
    def init_visual_stim(self, stimulus_key='', seed=1):
        """ all parameters have to be lumped in self.params """
        self.visual_stim = visual_stimulus(stimulus_key,
                                           stimulus_params=self.params,
                                           screen_params=self.params)

    def init_eye_movement(self, eye_movement_key, seed=1):
        """ all parameters have to be lumped in self.params """
        boundary_extent_limit = max(self.params['rf_size'])*self.params['convolve_extent_factor']
        self.eye_movement = virtual_eye_movement(eye_movement_key,
                                                 self.t_screen,
                                                 params=self.params,
                                                 boundary_extent_limit=boundary_extent_limit,
                                                 seed=seed,
                                                 screen_params=self.params)
        
        
    def setup_RF_props(self):

        # range of x-positions for the cellular RFs
        min_x0 = self.SCREEN['width']*self.params['rf_fraction']/2.+\
                    self.params['convolve_extent_factor']*self.params['rf_size'][1]
        max_x0 = self.SCREEN['width']-min_x0
        # range of y-positions for the cellular RFs
        min_y0 = self.SCREEN['height']*self.params['rf_fraction']/2.+\
                    self.params['convolve_extent_factor']*self.params['rf_size'][1]
        max_y0 = self.SCREEN['height']-min_y0
                                                       
        self.RF_PROPS = {
            'x0':[min_x0, max_x0],
            'y0':[min_y0, max_y0],
            'size':self.params['rf_size'],
            'freq':self.params['rf_freq'],
            'beta':self.params['rf_beta'],
            'theta':self.params['rf_theta'],
            'psi':self.params['rf_psi']
        }

        
    def screen_plot(self, array, **args):
        screen_plot(self.ge, array, self.SCREEN, **args)

        
    def draw_cell_RF_properties(self, seed,
                                clustered_features=True,
                                n_clustering=5):

        np.random.seed(int(seed))
        
        self.CELLS = {}

        for key, (vstart, vend) in self.RF_PROPS.items():

            if clustered_features:
                self.CELLS[key] = np.random.choice(np.linspace(vstart, vend, n_clustering),
                                              int(self.params['Ncells']))
            else:
                self.CELLS[key] = np.random.uniform(vstart, vend, size=int(self.params['Ncells']))



    def cell_gabor(self, i,
                   x_shift=0.,
                   y_shift=0.,
                   normalized=False,
                   width_factor=2., # to determine the center condition
                   with_center_conditions=False):

        gb = gabor(self.SCREEN['x_2d'], self.SCREEN['y_2d'],
                   x0=self.CELLS['x0'][i]+x_shift,
                   y0=self.CELLS['y0'][i]+y_shift,
                   freq=self.CELLS['freq'][i],
                   size=self.CELLS['size'][i],
                   beta=self.CELLS['beta'][i],
                   theta=self.CELLS['theta'][i],
                   psi=self.CELLS['psi'][i])

        if with_center_conditions:

            # find x-y boundaries, including the possible shift in visual space
            cond_x = (self.SCREEN['x_1d']>=self.CELLS['x0'][i]+x_shift-width_factor*self.CELLS['size'][i]) &\
                (self.SCREEN['x_1d']<self.CELLS['x0'][i]+x_shift+width_factor*self.CELLS['size'][i])
            cond_y = (self.SCREEN['y_1d']>=self.CELLS['y0'][i]+y_shift-width_factor*self.CELLS['size'][i]) &\
                (self.SCREEN['y_1d']<self.CELLS['y0'][i]+y_shift+width_factor*self.CELLS['size'][i])

            if normalized:
                # norm_factor = convolution_function(gb[cond_x, cond_y], gb[cond_x, cond_y])
                norm_factor = self.convolution_function(gb, gb)
            else:
                norm_factor = 1.


            if norm_factor>0:
                return gb/norm_factor, cond_x, cond_y
            else:
                return 0, cond_x, cond_y

        else:
            if normalized:
                norm_factor = self.convolution_function(gb, gb)
            else:
                norm_factor = 1.

            if norm_factor>0:
                return gb/norm_factor
            else:
                return 0
                
    def plot_RF_properties(self):
        
        Z = 0*self.SCREEN['x_2d']
        
        for i in range(self.params['Ncells']):
            z = self.cell_gabor(i)
            Z += z

        self.screen_plot(Z+0.5)

    ################################
    ### CONVOLUTION ################
    ################################

    # build_iterator
    def convolution_function(self, array1, array2):
        tot = 0
        for i, j in itertools.product(range(array1.shape[0]), range(array1.shape[1])):
            tot += array1[i,j]*array2[i,j]
        return tot

    def convol_func_gabor_restricted(self, array, icell,
                                     x_shift=0,
                                     y_shift=0):

        # compute gabor filter of cell icell, with the gaze-shift
        gb, cond_x, cond_y = self.cell_gabor(icell, 
                                             x_shift=x_shift, y_shift=y_shift,
                                             normalized=True,
                                             with_center_conditions=True)

        iterator = itertools.product(self.SCREEN['xd_1d'][cond_x], self.SCREEN['yd_1d'][cond_y])

        tot = 0
        for i, j in iterator:
            tot += gb[i,j]*array[i,j]
        return tot
    
        # if array.shape==gb.shape:
        #     tot = 0
        #     for i, j in iterator:
        #         tot += gb[i,j]*array[i,j]
        # elif array[0, :, :].shape==gb.shape:
        #     tot = np.zeros(array.shape[0])
        #     for it in len(tot):
        #         for i, j in iterator:
        #             tot[it] += gb[i,j]*array[it, i, j]
        # else:
        #     print('unable to match array shape')
        #     tot=None
        # return tot

    def RF_filtering(self, icell_range='all'):
        """
        needs to pass visual_stim and eye_movement objects from stimli.py and virtual_eye_movement.py

        based on t_screen, no need to have it faster that this
        """
        if icell_range is 'all':
            icell_range = np.arange(self.params['Ncells'], dtype=int)
            
        RF_filtered = np.zeros((len(icell_range), len(self.t_screen)))
        
        if (self.visual_stim is None) or (self.eye_movement is None):
            print("""
            /!\ Need to instantiate "visual_stim" and "eye_movement" to get a RF response
                     --> returning null activity
            """)
        else: #
            print('[...] Performing RF filtering of the visual input')
            for it, tt in enumerate(self.t_screen):
                vis = self.visual_stim.get(tt)
                em_x, em_y = self.eye_movement.x[it], self.eye_movement.y[it]
                for icell in icell_range:
                    RF_filtered[icell, it] = self.convol_func_gabor_restricted(vis, icell, em_x, em_y)
                
        return RF_filtered

    def save_RF_filtered_data(self, filename, RF_filtered):

        data = self.params
        data['CELLS'] = self.CELLS
        data['t_screen'] = self.t_screen
        data['RF_filtered'] = RF_filtered

        if '.npz' not in filename:
            np.savez(filename+'.npz', **data)
        else:
            np.savez(filename, **data)

        
    #################################
    ### LNP MODEL IMPLEMENTATION ####
    #################################
    
    def operator_delay_adapt(self, s, a, r):
        """
        r[i+1= r[i]+dt/tau_delay*(-r[i]+s[i]-a[i])
        a[i]+dt/tau_adapt*((1-fraction_adapt)/fraction_adapt*r[i]-a[i])
        """
        return [r+self.model['dt']/self.model['tau_delay']*(s-r-a),
                a+self.model['dt']/self.model['tau_adapt']*((1-self.model['fraction_adapt'])/self.model['fraction_adapt']*r-a)]

    def temporal_filtering(self, t, input_signal):
        if (t[1]-t[0])!=model['dt']:
            print('/!\ time step not accurate in the simulation !!')
        r, a = 0*t, 0*t
        for i in range(len(t)-1):
            r[i+1], a[i+1] = self.operator_delay_adapt(input_signal[i],
                                                       a[i], r[i])
        return r, a

            
    def NL_function(self, x):
        x[x<=self.model['NL_threshold']] = self.model['NL_threshold']
        return self.model['NL_slope_Hz_per_Null']*x

    def compute_rates(self, LIN_OUTPUT):
        return [self.NL_function(lo) for lo in LIN_OUTPUT]

    def Poisson_process_transform(self, t, RATES, seed=0):
        """
        inhomogeneous POisson process
        """
        np.random.seed(seed)
        RATES = np.array(RATES)
        RDM = np.random.uniform(0, 1, size=RATES.shape)
        SPIKES = [[] for i in range(RATES.shape[0])]

        for (n, it) in np.argwhere(RDM<RATES*dt):
            SPIKES[n].append(t[it])
        return SPIKES

        
if __name__=='__main__':

    model = earlyVis_model(graph_env_key='manuscript')
    
    # model.draw_cell_RF_properties(3, clustered_features=False)

    # model.init_visual_stim('drifting-grating')
    # model.init_eye_movement('saccadic')

    # RF = model.RF_filtering()
    # model.save_RF_filtered_data('data.npz', RF)

    data = np.load('data.npz')
    
    # # model.plot_RF_properties()
    # model.ge.plot(data['t_screen'], Y=data['RF_filtered'], fig_args={'figsize':(3,1)})

    from scipy.interpolate import interp1d
    new_t = np.linspace(data['t_screen'][0], data['t_screen'][-1], 1000)
    def extrapolate_RF_trace(trace, old_t, new_t):
        interp = interp1d(old_t, trace, kind='quadratic')
        return interp(new_t)
    RF2 = [extrapolate_RF_trace(rf, data['t_screen'], new_t) for rf in data['RF_filtered']]
    fig, ax = model.ge.figure(figsize=(3,1))
    model.ge.plot(data['t_screen'], Y=data['RF_filtered'], ax=ax)
    model.ge.plot(new_t, Y=RF2, ax=ax)
    model.ge.show()





