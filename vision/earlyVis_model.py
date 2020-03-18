import numpy as np

import itertools, string, sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from datavyz.main import graph_env
from vision.gabor_filters import gabor


params0 = {
    #
    'Ncells':100,
    # units of the visual field is degree
    'screen_width':16./9.*30, # degree
    'screen_height':30.,
    'screen_dpd':5, # dot per degree (dpd)
    'screen_refresh_rate':30., #in Hz
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
    # numerical simulations
    'dt':10e-3,
    'tstop':1.,
}

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
            self.params = params0 # above params by default

        self.setup_screen()
        self.setup_RF_props()
        
        self.ge = graph_env(graph_env_key)

        
    def setup_screen(self):

        SCREEN = {}
        # in dots
        SCREEN['Xd_max'] = int(self.params['screen_width']*self.params['screen_dpd'])
        SCREEN['Yd_max'] = int(self.params['screen_height']*self.params['screen_dpd'])
        SCREEN['xd_1d'], SCREEN['yd_1d'] = np.arange(SCREEN['Xd_max']), np.arange(SCREEN['Yd_max'])
        SCREEN['xd_2d'], SCREEN['yd_2d'] = np.meshgrid(SCREEN['xd_1d'], SCREEN['yd_1d'], indexing='ij')
        # in degrees
        SCREEN['x_1d'] = SCREEN['xd_1d']/self.params['screen_dpd']
        SCREEN['y_1d'] = SCREEN['yd_1d']/self.params['screen_dpd']
        SCREEN['x_2d'], SCREEN['y_2d'] = np.meshgrid(SCREEN['x_1d'], SCREEN['y_1d'], indexing='ij')
        self.SCREEN = SCREEN


    def setup_RF_props(self):

        # range of x-positions for the cellular RFs
        min_x0 = self.params['screen_width']*self.params['rf_fraction']/2.+\
                    self.params['convolve_extent_factor']*self.params['rf_size'][1]
        max_x0 = self.params['screen_width']-min_x0
        # range of y-positions for the cellular RFs
        min_y0 = self.params['screen_height']*self.params['rf_fraction']/2.+\
                    self.params['convolve_extent_factor']*self.params['rf_size'][1]
        max_y0 = self.params['screen_height']-min_y0
                                                       
        self.RF_PROPS = {
            'x0':[min_x0, max_x0],
            'y0':[min_y0, max_y0],
            'size':self.params['rf_size'],
            'freq':self.params['rf_freq'],
            'beta':self.params['rf_beta'],
            'theta':self.params['rf_theta'],
            'psi':self.params['rf_psi']
        }
        
    def screen_plot(self, array,
                    ax=None,
                    Ybar_label='',
                    Xbar_label='10$^o$'):
        if ax is None:
            fig, ax = self.ge.figure()
        else:
            fig = None
        self.ge.twoD_plot(self.SCREEN['x_2d'].flatten(),
                          self.SCREEN['y_2d'].flatten(),
                          array.flatten(),
                          vmin=0, vmax=1,
                          colormap=self.ge.binary,
                          ax=ax)
        self.ge.draw_bar_scales(ax,
                                Xbar=10., Ybar_label=Ybar_label,
                                Ybar=10., Xbar_label=Xbar_label,
                                xyLoc=(-0.02*self.params['screen_width'],
                                       1.02*self.params['screen_height']),
                                loc='left-top')
        ax.axis('equal')
        ax.axis('off')
        return fig, ax
        

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


            return gb/norm_factor, cond_x, cond_y

        else:
            if normalized:
                norm_factor = self.convolution_function(gb, gb)
            else:
                norm_factor = 1.

            return gb/norm_factor
                
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

        if array.shape==gb.shape:
            tot = 0
            for i, j in iterator:
                tot += gb[i,j]*array[i,j]
        elif array[0, :, :].shape==gb.shape:
            tot = np.zeros(array.shape[0])
            for it in len(tot):
                for i, j in iterator:
                    tot[it] += gb[i,j]*array[it, i, j]
        else:
            print('uable to match array shape')
            tot=None

        return tot

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
    model = earlyVis_model()
    model.draw_cell_RF_properties(3, clustered_features=False)
    model.plot_RF_properties()
    model.ge.show()





