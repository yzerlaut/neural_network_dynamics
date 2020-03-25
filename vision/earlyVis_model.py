import itertools, string, sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.interpolate import interp1d
    
from analyz.IO.npz import save_dict, load_dict

from vision.gabor_filters import gabor
from vision.stimuli import setup_screen, screen_params0, stim_params0, visual_stimulus
from vision.virtual_eye_movement import virtual_eye_movement, vem_params0

params0 = {
    # visual space props
    'height_VF':40, # degree, Angular height of the visual field
    'width_VF':int(round(10*16./9.*40,0)/10), # degree, Angular width of the visual field
    'center_VF':45, # degree, Field center from antero-posterior axis
    # neuronal space props
    'Ncells':100,
    'Area_cells': np.round(np.sqrt(100/177e3/0.2),2),#sqrt(Ncell/177e3/Height_L4),177e3->Markram (2015)
    # receptive fields
    'clustered_features':True,
    'rf_fraction':.4, # fraction of visual space covered by the cells, fraction of the screen 
    'rf_x0':[20, 70.], # degrees
    'rf_y0':[20, 30.], # degrees
    'rf_size':[2., 8.], # degrees
    'rf_freq':[0.02, 0.12], # cycle per degrees
    'rf_beta':[0.8, 2.5],
    'rf_theta':[0., np.pi],
    'rf_psi':[0., 2*np.pi], 'rf_psi_peak1':np.pi, 'rf_psi_peak2':3*np.pi/2, 'rf_psi_Dwidth':np.pi/4,
    'convolve_extent_factor':1.5, # limit the convolution to thisfactor*rf-width to make comput faster
    # temporal filtering
    'tau_adapt':500e-3,
    'tau_delay':30e-3,
    'fraction_adapt':0.2,
    # non-linear amplification
    'NL_baseline':0.5,
    'NL_slope_Hz_per_Null':20.,
    # # virtual eye movement
    'duration_distance_slope':1.9e-3, # degree/s
    'duration_distance_shift':63e-3, # s
    # simulation params
    'dt':10e-3,
    'tstop':2.3,
}

full_params0 = {**params0, **screen_params0, **stim_params0, **vem_params0}
np.savez('docs/params.npz', **full_params0)

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
                 from_file=None,
                 params=None):


        if from_file is not None:
            self.load_data(from_file)
        elif params is not None:
            self.params = params
        else:
            self.params = full_params0 # above params by default

        self.Ncells = self.params['Ncells']
        self.SCREEN = setup_screen(self.params)
        
        self.setup_RF_props()

        self.visual_stim = None
        self.dt_screen = 1./self.params['screen_refresh_rate']
        self.t_screen = np.arange(int(self.params['tstop']/self.dt_screen)+1)*self.dt_screen
        
        self.eye_movement = None
        
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
        # min_x0 = self.SCREEN['width']*self.params['rf_fraction']/2.+\
        #             self.params['convolve_extent_factor']*self.params['rf_size'][1]
        min_x0 = self.params['convolve_extent_factor']*self.params['rf_size'][1]
        max_x0 = self.SCREEN['width']-min_x0
        if (min_x0>self.params['rf_x0'][0]) or (max_x0<self.params['rf_x0'][1]):
            print('--------------------------------------------------------------------------')
            print('/!\ x0 bounds (%.1f, %.1f) do not allow to perform the convolution on the full screen !' % (self.params['rf_x0'][0], self.params['rf_x0'][1]))
            print('/!\ x0 bounds do not allow to perform the convolution on the full screen !')
            print(' you should provide a value range within than:', (min_x0, max_x0))
        # range of y-positions for the cellular RFs
        min_y0 = self.params['convolve_extent_factor']*self.params['rf_size'][1]
        max_y0 = self.SCREEN['height']-min_y0
        if (min_y0>self.params['rf_y0'][0]) or (max_y0<self.params['rf_y0'][1]):
            print('--------------------------------------------------------------------------')
            print('/!\ y0 bounds (%.1f, %.1f) do not allow to perform the convolution on the full screen !' % (self.params['rf_y0'][0], self.params['rf_y0'][1]))
            print('/!\ y0 bounds do not allow to perform the convolution on the full screen !')
            print(' you should provide a value range within than:', (min_y0, max_y0))
                                                       
        self.RF_PROPS = {
            'x0':self.params['rf_x0'],
            'y0':self.params['rf_y0'],
            'size':self.params['rf_size'],
            'freq':self.params['rf_freq'],
            'beta':self.params['rf_beta'],
            'theta':self.params['rf_theta'],
            'psi':self.params['rf_psi']
        }

        
        
    def draw_cell_RF_properties(self, seed,
                                clustered_features=True,
                                n_clustering=10):

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

    def keys_for_saving(self):
        return ['CELLS', 't_screen', 'EM', 'RF', 'ADAPT', 'RATES', 'SPIKES']
    
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
        # return [r+dt/self.params['tau_delay']*((np.sign(s)+1)*s/2.)-r-a,
        return [r+dt/self.params['tau_delay']*((np.sign(s)+1)*s/2.-r-a),
                a+dt/self.params['tau_adapt']*((1-self.params['fraction_adapt'])/self.params['fraction_adapt']*r-a)]

    def temporal_filtering(self, t, input_signal):

        dt = t[1]-t[0]
        r, a = 0*t, 0*t
        for i in range(len(t)-1):
            r[i+1], a[i+1] = self.operator_delay_adapt(input_signal[i], a[i], r[i], dt)
        return r, a


    def compute_rates(self, x):
        return self.params['NL_baseline']+self.params['NL_slope_Hz_per_Null']*x
 
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
        self.draw_cell_RF_properties(self.Ncells,
                                      clustered_features=self.params['clustered_features'])

        self.init_visual_stim(stimulus_key, seed=seed+1)
        self.init_eye_movement(eye_movement_key, seed=seed+2)
        self.RF_filtering()
        
    def half_process2(self,
                      seed=3):
        """
        from RF traces to spikes
        """
        RF0 = self.resample_RF_traces()
        self.RF, self.ADAPT = 0*RF0, 0*RF0
        self.RATES = 0*RF0
        print('[...] Temporal filtering of the RF-traces (delay and adaptation)')
        for icell in range(self.Ncells):
            self.RF[icell,:], self.ADAPT[icell,:] = self.temporal_filtering(self.t, RF0[icell,:])
            
        print('[...] Non-linear transformation of RF-traces to get firing rates')
        for icell in range(self.Ncells):
            self.RATES[icell,:] = self.compute_rates(self.RF[icell,:])

        self.Poisson_process_transform(seed+1)

        
    def full_process(self, stimulus_key, eye_movement_key, seed=2):
        """
        full process of the model
        """
        self.half_process1(stimulus_key, eye_movement_key, seed=seed)
        self.half_process2(seed=seed+1)

    def save_data(self, filename):

        data = self.params
        for key in self.keys_for_saving():
            data[key] = getattr(self, key)
        save_dict(filename, data)

    def load_data(self, filename):

        data = load_dict(filename)

        self.params = {}
        for key, val in data.items():
            if key in self.keys_for_saving():
                setattr(self, key, val)
            else:
                self.params[key] = val

        
if __name__=='__main__':

    model = earlyVis_model()

    # model.full_process('drifting-grating', 'saccadic', seed=3)
    # model.save_data('data/drifting-grating-saccadic')

    # model.full_process('drifting-grating', 'saccadic', seed=3)
    model.load_data('data/drifting-grating-saccadic.npz')
    
    # # model.full_process('drifting-grating', '', seed=3)
    
    # model.half_process1('sparse-noise', 'saccadic', seed=3)
    # model.save_RF_filtered_data('data.npz')
    
    # model.load_RF_filtered_data('data.npz')
    # model.half_process2()
    
    # fig = model.protocol_plot()
    # fig.savefig('docs/fig2.png')
    # model.ge.show()
