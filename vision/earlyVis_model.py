import itertools, string, sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pylab as plt
from scipy.interpolate import interp1d
    
from datavyz.main import graph_env
from analyz.IO.npz import save_dict, load_dict

from vision.gabor_filters import gabor
from vision.stimuli import setup_screen, screen_plot, screen_params0, stim_params0, visual_stimulus
from vision.virtual_eye_movement import virtual_eye_movement, vem_params0

params0 = {
    #
    'Ncells':4,
    # receptive fields
    'clustered_features':True,
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
    'tstop':2.3,
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
                 graph_env_key='manuscript'):
        
        if params is not None:
            self.params = params
        else:
            self.params = full_params0 # above params by default

        self.Ncells = self.params['Ncells']
        self.SCREEN = setup_screen(self.params)
        
        self.setup_RF_props()

        self.visual_stim = None
        self.dt_screen = 1./self.params['screen_refresh_rate']
        self.t_screen = np.arange(int(self.params['tstop']/self.dt_screen)+1)*self.dt_screen
        
        self.eye_movement, self.EM = None, {}
        
        self.ge = graph_env(graph_env_key)

        self.dt = self.params['dt']
        self.t = np.arange(int(self.params['tstop']/self.dt))*self.dt

        
    def init_visual_stim(self, stimulus_key='', seed=1):
        """ all parameters have to be lumped in self.params """
        print('[...] Initializing the visual stimulation')
        self.params['stimulus_key'] = stimulus_key
        self.visual_stim = visual_stimulus(stimulus_key,
                                           stimulus_params=self.params,
                                           screen_params=self.params)

    def init_eye_movement(self, eye_movement_key, seed=1):
        """ all parameters have to be lumped in self.params """
        print('[...] Initializing eye movement')
        boundary_extent_limit = max(self.params['rf_size'])*self.params['convolve_extent_factor']
        self.params['boundary_extent_limit'] = boundary_extent_limit
        self.params['eye_movement_key'] = eye_movement_key
        self.eye_movement = virtual_eye_movement(eye_movement_key,
                                                 self.t_screen,
                                                 params=self.params,
                                                 boundary_extent_limit=boundary_extent_limit,
                                                 seed=seed,
                                                 screen_params=self.params)
        self.EM = {'x':self.eye_movement.x,
                   'y':self.eye_movement.y,
                   'events':self.eye_movement.events}
        
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

    def protocol_plot(self, Ncell_plot=3):

        fig, AX = self.ge.figure(axes_extents=[[[1,2], [1,2], [1,2], [1,2], [1,2]],
                                               [[5,1]],
                                               [[5,2]],
                                               [[5,2]],
                                               [[5,2]],
                                               [[5,2]],
                                               [[5,2]],
                                               [[5,2]]],
                                 figsize=(.7, .25),
                                 hspace=0.5, top=0.8, left=1.2)

        visual_stim = visual_stimulus(self.params['stimulus_key'],
                                      stimulus_params=self.params,
                                      screen_params=self.params)

        for i in range(5):
            it=int(self.t[-1]/self.dt/5)*i
            its=int(self.t_screen[-1]/self.dt_screen/5)*i
            screen_plot(self.ge, visual_stim.get(self.t[it]), self.SCREEN,
                        Xbar_label='10$^o$  t=%.1fs' % self.t[it], ax=AX[0][i])
            self.ge.multicolored_line(self.EM['x']*self.params['screen_dpd'],
                                      self.EM['y']*self.params['screen_dpd'],
                                      np.linspace(0, 1, len(self.EM['x'])),
                                      ax=AX[0][i], lw=0.5)
            AX[0][i].scatter([self.EM['x'][its]*self.params['screen_dpd']],
                             [self.EM['y'][its]*self.params['screen_dpd']],
                             alpha=1, s=10,
                             color=self.ge.cool(np.linspace(0, 1, len(self.EM['x']))[its]))

        AX[1][0].axis('off')
        upper_plot_pos = AX[2][0].get_position()
        time_axis = plt.axes([upper_plot_pos.x0, upper_plot_pos.y1,
                              upper_plot_pos.x1-upper_plot_pos.x0,
                              .2*(upper_plot_pos.y1-upper_plot_pos.y0)])
        self.ge.multicolored_line(self.t_screen, np.zeros(len(self.t_screen)),
                                  np.linspace(0.3, 1., len(self.t_screen)),
                                  ax=time_axis, lw=3)
        self.ge.annotate(time_axis, 'time (s)', (.5, 1.), ha='center')
        self.ge.set_plot(time_axis, [], xlim=[self.t[0], self.t[-1]], ylim=[-1,1])

        ixc, iyc = int(self.SCREEN['Xd_max']/2),int(self.SCREEN['Yd_max']/2)
        luminance_at_center = [visual_stim.get(t)[ixc, iyc] for t in self.t_screen]
        AX[2][0].plot(self.t_screen, luminance_at_center)
        self.ge.set_plot(AX[2][0], ['left', 'top'], xlim=[self.t[0], self.t[-1]],
                         ylabel='norm. lum.\nat center')

        AX[3][0].plot(self.t_screen, self.EM['x']-self.EM['x'][0]+1, color='firebrick')
        self.ge.annotate(AX[3][0], 'x(t)', (0.,4), xycoords='data', color='firebrick')
        AX[3][0].plot(self.t_screen, self.EM['y']-self.EM['y'][0]-1, color='olivedrab')
        self.ge.annotate(AX[3][0], 'y(t)', (0.,-4),xycoords='data',color='olivedrab', va='top')
        self.ge.set_plot(AX[3][0], ['left'], ylabel='eye \nmov.($^{o}$)',
                         xlim=[self.t[0], self.t[-1]],
                         ylim=.5*self.SCREEN['width']*np.array([-1,1]))

        # loop over cells:
        for i in range(Ncell_plot):
            plot_pos = AX[4+i][0].get_position()
            inset = plt.axes([plot_pos.x0, plot_pos.y0+.55*(plot_pos.y1-plot_pos.y0),
                              16./9.*.45*(plot_pos.y1-plot_pos.y0),
                              .45*(plot_pos.y1-plot_pos.y0)])
            z = self.cell_gabor(i)
            inset.axis('off')
            self.ge.matrix(z,
                           vmin=-np.abs(z).max(), vmax=np.abs(z).max(),
                           colormap=self.ge.binary_r,
                           bar_legend=None,
                           ax=inset)
            AX[4+i][0].plot(self.t_screen, self.RF_filtered[i,:])
            AX[4+i][0].plot(self.t, self.RATES[i,:], lw=3)
            self.ge.set_plot(AX[4+i][0], ['left'], ylabel='cell %i' % (i+1), yticks=[])

        for i, spk in enumerate(self.SPIKES):
            AX[-1][0].scatter(spk, i*np.ones(len(spk)), s=3, color=self.ge.brown)
        self.ge.set_plot(AX[-1][0], ylabel='cell ID', xlabel='time (s)')
            
        
    def draw_cell_RF_properties(self, seed,
                                clustered_features=True,
                                n_clustering=5):

        np.random.seed(int(seed))
        
        self.CELLS = {}

        for key, (vstart, vend) in self.RF_PROPS.items():

            if clustered_features:
                self.CELLS[key] = np.random.choice(np.linspace(vstart, vend, n_clustering),
                                              int(self.Ncells))
            else:
                self.CELLS[key] = np.random.uniform(vstart, vend, size=int(self.Ncells))



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
        
        for i in range(self.Ncells):
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
    

    def RF_filtering(self, icell_range='all'):
        """
        needs to pass visual_stim and eye_movement objects from stimli.py and virtual_eye_movement.py

        based on t_screen, no need to have it faster that this
        """
        if icell_range is 'all':
            icell_range = np.arange(self.Ncells, dtype=int)
            
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
                em_x, em_y = self.EM['x'][it], self.EM['y'][it]
                for icell in icell_range:
                    RF_filtered[icell, it] = self.convol_func_gabor_restricted(vis, icell, em_x, em_y)
                    
        self.RF_filtered = RF_filtered

    def intermediate_keys_for_saving(self):
        return ['CELLS', 't_screen', 'RF_filtered', 'EM']
    
    def save_RF_filtered_data(self, filename):

        data = self.params
        for key in self.intermediate_keys_for_saving():
            data[key] = getattr(self, key)
        save_dict(filename, data)

    def load_RF_filtered_data(self, filename):

        data = load_dict(filename)

        self.params = {}
        for key, val in data.items():
            if key in self.intermediate_keys_for_saving():
                setattr(self, key, val)
            else:
                self.params[key] = val

        
    def resample_RF_traces(self):

        print('[...] Resampling RF traces on high-temporal-res axis (linear interpol.)')
        new_RF_filtered = np.zeros((self.RF_filtered.shape[0], len(self.t)))
        for icell in range(self.RF_filtered.shape[0]):
            interp = interp1d(self.t_screen, self.RF_filtered[icell, :], kind='linear')
            new_RF_filtered[icell,:] = interp(self.t)

        return new_RF_filtered

        
    #################################
    ### LNP MODEL IMPLEMENTATION ####
    #################################
    
    def operator_delay_adapt(self, s, a, r, dt):
        """
        r[i+1= r[i]+dt/tau_delay*(-r[i]+s[i]-a[i])
        a[i]+dt/tau_adapt*((1-fraction_adapt)/fraction_adapt*r[i]-a[i])
        """
        return [r+dt/self.params['tau_delay']*(s-r-a),
                a+dt/self.params['tau_adapt']*((1-self.params['fraction_adapt'])/self.params['fraction_adapt']*r-a)]

    def temporal_filtering(self, t, input_signal):

        dt = t[1]-t[0]
        r, a = 0*t, 0*t
        for i in range(len(t)-1):
            r[i+1], a[i+1] = self.operator_delay_adapt(input_signal[i], a[i], r[i], dt)
        return r, a


    def compute_rates(self, x):
        cond = (x<=self.params['NL_threshold'])
        x[cond] = self.params['NL_threshold']
        return self.params['NL_slope_Hz_per_Null']*x
 
    def Poisson_process_transform(self, seed=0):
        """
        inhomogeneous POisson process
        """
        print('[...] Transforming rates into spikes (inhomogenous Poisson process hyp.)')
        np.random.seed(seed)
        RATES = np.array(self.RATES)
        RDM = np.random.uniform(0, 1, size=RATES.shape)
        self.SPIKES = [[] for i in range(RATES.shape[0])]

        for (n, it) in np.argwhere(RDM<RATES*self.dt):
            self.SPIKES[n].append(self.t[it])


    def half_process1(self, stimulus_key, eye_movement_key,
                      seed=2):
        """
        from drawing to the traces from the spatial filtering of the visual input
        """
        model.draw_cell_RF_properties(model.Ncells,
                                      clustered_features=model.params['clustered_features'])

        model.init_visual_stim(stimulus_key, seed=seed+1)
        model.init_eye_movement(eye_movement_key, seed=seed+2)

        model.RF_filtering()
        
    def half_process2(self,
                      seed=3):
        """
        from RF traces to spikes
        """
        RF = model.resample_RF_traces()
        RF_TF, RF_ADAPT = 0*RF, 0*RF
        self.RATES = 0*RF
        print('[...] Temporal filtering of the RF-traces (delay and adaptation)')
        for icell in range(model.Ncells):
            RF_TF[icell,:], RF_ADAPT[icell,:] = model.temporal_filtering(model.t, RF[icell,:])
            
        print('[...] Non-linear transformation of RF-traces to get firing rates')
        for icell in range(model.Ncells):
            self.RATES[icell,:] = model.compute_rates(RF_TF[icell,:])

        self.Poisson_process_transform(seed+1)

        
        
    def full_process(self, stimulus_key, eye_movement_key, seed=2):
        """
        full process of the model
        """
        model.half_process1(stimulus_key, eye_movement_key, seed=seed)
        # model.save_RF_filtered_data('data.npz')
        # model.load_RF_filtered_data('data.npz')
        model.half_process2(seed=seed+1)

        
if __name__=='__main__':

    model = earlyVis_model(graph_env_key='manuscript')

    # model.full_process('drifting-grating', 'saccadic', seed=3)
    # model.full_process('drifting-grating', '', seed=3)
    
    # model.half_process1('drifting-grating', '', seed=3)
    # model.save_RF_filtered_data('data.npz')
    
    model.load_RF_filtered_data('data.npz')
    model.half_process2()
    
    model.protocol_plot()
    model.ge.show()
