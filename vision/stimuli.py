import os, sys
import numpy as np
from scipy.interpolate import RectBivariateSpline, interp2d

import itertools, string, sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from datavyz.images import load
from analyz.signal_library.classical_functions import gaussian_2d

from vision.gabor_filters import gabor

screen_params0 = {
    # units of the visual field is degree
    'screen_width':16./9.*50, # degree
    'screen_height':50.,
    'screen_dpd':10, # dot per degree (dpd)
    'screen_refresh_rate':30., #in Hz
}

stim_params0 = {
    'stim_tstart':0.2, # in seconds
    'stim_tend':5, # in seconds
    'lumin_value_at_init':0.5, # grey screen in pre-screen
    'static':False,
    # gratings drifting and static
    'theta':np.pi/6.,
    'drifting_speed_cycle_per_second': 2.,
    'spatial_phase':np.pi,
    'spatial_freq':0.07,
    # center and surround props
    'center':(40, 20), 
    'center_radius':5,
    'center_contrast':1.,
    'center_spatial_freq' : 0.15,
    'center_spatial_phase' : np.pi,
    'center_theta' : np.pi/6.,
    'surround_radius':10,
    'surround_contrast':1.,
    'surround_spatial_freq' : 0.15,
    'surround_spatial_phase' : np.pi,
    'surround_theta' : np.pi/2.,
    # sparse noise
    'SN_noise_mean_refresh_time':0.3, # in s
    'SN_noise_rdm_jitter_refresh_time':0.1, # in s
    'SN_square_size':5., # in degrees
    'SN_sparseness':0.05,
    # sparse noise
    'DN_noise_mean_refresh_time':0.3, # in s
    'DN_noise_rdm_jitter_refresh_time':0.1, # in s
    'DN_square_size':5., # in degrees
    # gaussian blob & appearance
    'blob_center':[30.,20.],
    'blob_size':[8.,8.],
    'blob_amplitude':1.,
    'blob_rise_time':.4,
    'blob_time':0.5,
    # center-surround protocols
    'center_delay':0.3,
    'center_duration':.5,
    'surround_delay':0.,
    'surround_duration':2.,
    # natural images
    'NI_directory':'/home/yann/Pictures/Imagenet/',
}

def img_after_hist_normalization(img):
    """
    for NATURAL IMAGES:
    histogram normalization to get comparable images
    """
    print('2) Performing histogram normalization [...]')
    flat = np.array(1000*img.flatten(), dtype=int)

    cumsum = np.cumsum(np.histogram(flat, bins=np.arange(1001))[0])

    norm_cs = np.concatenate([(cumsum-cumsum.min())/(cumsum.max()-cumsum.min())*1000, [1]])
    new_img = np.array([norm_cs[f]/1000. for f in flat])

    return new_img.reshape(img.shape)

def adapt_to_screen_resolution(img, SCREEN):

    print('1) Adapting image to chosen screen resolution [...]')
    
    old_X = np.linspace(0, SCREEN['width'], img.shape[0])
    old_Y = np.linspace(0, SCREEN['height'], img.shape[1])
    
    new_X = np.linspace(0, SCREEN['width'], SCREEN['Xd_max'])
    new_Y = np.linspace(0, SCREEN['height'], SCREEN['Yd_max'])

    new_img = np.zeros((SCREEN['Xd_max'], SCREEN['Yd_max']))

    spline_approx = interp2d(old_X, old_Y, img.T, kind='linear')
    
    return spline_approx(new_X, new_Y).T
    
    
def setup_screen(params):
    """
    calculate the quantities in degrees and in dots to handle screen display
    """
    SCREEN = {'width':params['screen_width'],
              'height':params['screen_height'],
              'dpd':params['screen_dpd'],
              'refresh_rate':params['screen_refresh_rate'], # in Hz
    }
    
    # in dots
    SCREEN['Xd_max'], SCREEN['Yd_max'] = int(SCREEN['width']*SCREEN['dpd']), int(SCREEN['height']*SCREEN['dpd'])
    SCREEN['Xd_center'], SCREEN['Yd_center'] = int(SCREEN['width']*SCREEN['dpd']/2), int(SCREEN['height']*SCREEN['dpd']/2)
    SCREEN['xd_1d'], SCREEN['yd_1d'] = np.arange(SCREEN['Xd_max']), np.arange(SCREEN['Yd_max'])
    SCREEN['xd_2d'], SCREEN['yd_2d'] = np.meshgrid(SCREEN['xd_1d'], SCREEN['yd_1d'], indexing='ij')
    
    # in degrees
    SCREEN['x_1d'] = SCREEN['xd_1d']/SCREEN['dpd']
    SCREEN['y_1d'] = SCREEN['yd_1d']/SCREEN['dpd']
    SCREEN['x_2d'], SCREEN['y_2d'] = np.meshgrid(SCREEN['x_1d'], SCREEN['y_1d'], indexing='ij')
    SCREEN['x_max'], SCREEN['y_max'] = SCREEN['Xd_max']/SCREEN['dpd'], SCREEN['Yd_max']/SCREEN['dpd']
    SCREEN['x_center'], SCREEN['y_center'] = SCREEN['Xd_center']/SCREEN['dpd'], SCREEN['Yd_center']/SCREEN['dpd']
    
    return SCREEN


class visual_stimulus:

    def __init__(self,
                 stimulus_key='',
                 stimulus_params=None,
                 screen_params=None,
                 stim_number=1,
                 seed=1):

        if screen_params is not None:
            self.screen_params = screen_params
        else:
            self.screen_params = screen_params0 # above params by default

        if stimulus_params is not None:
            self.stimulus_params = stimulus_params
        else:
            self.stimulus_params = stim_params0
            
        self.SCREEN = setup_screen(self.screen_params)
        
        self.initialize_screen_time_axis()
        
        if stimulus_key=='static-full-field-grating':
            self.static_full_field_grating(**self.stimulus_params)
        elif stimulus_key=='static-center-grating':
            self.static_center_grating(**self.stimulus_params)
        elif stimulus_key=='static-surround-grating':
            self.static_surround_grating(**self.stimulus_params)
        elif stimulus_key=='static-center-surround-grating':
            self.static_center_surround_grating(**self.stimulus_params)
        elif stimulus_key=='natural-images':
            self.natural_images(image_number=stim_number)
        elif stimulus_key in ['drifting-grating', 'full-field-drifting-grating']:
            self.drifting_grating(**self.stimulus_params)
        elif stimulus_key=='sparse-noise':
            self.sparse_noise(seed=seed, **self.stimulus_params)
        elif stimulus_key=='dense-noise':
            self.dense_noise(seed=seed, **self.stimulus_params)
        elif stimulus_key in ['gaussian-blob-appearance', 'gaussian-blob']:
            # self.stimulus_params['lumin_value_at_init']=0., # black screen in pre-screen
            self.gaussian_blob_appearance(**self.stimulus_params)
        elif stimulus_key=='gaussian-blob':
            self.gaussian_blob_static(**self.stimulus_params)
        elif stimulus_key in ['center-surround-protocols', 'center-surround']:
            print('ok !!')
            self.center_surround_protocols(**self.stimulus_params)
        elif stimulus_key=='black-screen': # grey screen by default
            self.black_screen()
        elif stimulus_key=='grey-screen': # grey screen by default
            self.grey_screen()
        elif stimulus_key=='white-screen': # grey screen by default
            self.white_screen()
        else: # grey screen by default
            self.grey_screen()

    def initialize_screen_time_axis(self):
        """
        initialize time array based on refresh rate
        """
        self.t0 = self.stimulus_params['stim_tstart']
        self.tstop = self.stimulus_params['stim_tend']
        self.iTmax = int(self.tstop*self.SCREEN['refresh_rate'])+1
        self.it0 = int(self.t0*self.SCREEN['refresh_rate'])+1
        self.screen_time_axis = np.arange(self.iTmax)/self.SCREEN['refresh_rate']
        self.screen_time_axis_from_t0 = self.screen_time_axis[self.it0:]-self.t0

    def from_time_to_array_index(self, t):
        if self.stimulus_params['static'] and (t<=self.t0):
             return 0
        elif self.stimulus_params['static']:
            return 1
        else:
            return int(t*self.SCREEN['refresh_rate'])
    
    def get(self, t):
        """
        returns the 
        """
        return self.full_array[self.from_time_to_array_index(t), :, :]
    
    ###################################################################
    ################### SET OF DIFFERENT STIMULI ######################
    ###################################################################

    # dynamic
    def initialize_dynamic(self):
        self.stimulus_params['static'] = False
        # index [0,:,:] is pre-stim, [1:,:,:] is stim
        self.full_array = np.ones((len(self.screen_time_axis),
                                   self.SCREEN['Xd_max'],
                                   self.SCREEN['Yd_max']))*self.stimulus_params['lumin_value_at_init']
        # then only need to fill > self.t0 !

    # static
    def initialize_static(self):

        self.stimulus_params['static'] = True
        # index [0,:,:] is pre-stim, [1,:,:] is stim
        self.full_array = np.ones((2, self.SCREEN['Xd_max'],
                                   self.SCREEN['Yd_max']))*self.stimulus_params['lumin_value_at_init']

        
    # ------------------
    # static stimuli
    # ------------------
    
    def static_full_field_grating(self,
                                  i_screen_time=1, # >= onset for static stim
                                  spatial_freq = 0.07,
                                  spatial_phase = np.pi,
                                  contrast=1.,
                                  theta = np.pi/6., **args):

        if i_screen_time==1:
            self.initialize_static()
        
        x_theta = (self.SCREEN['x_2d']-self.SCREEN['x_center']) * np.cos(theta) + (self.SCREEN['y_2d']-self.SCREEN['y_center']) * np.sin(theta)
        y_theta = -(self.SCREEN['x_2d']-self.SCREEN['x_center']) * np.sin(theta) + (self.SCREEN['y_2d']-self.SCREEN['y_center']) * np.cos(theta)

        self.full_array[i_screen_time,:,:] = 0.5-contrast*np.cos(2*np.pi*spatial_freq*x_theta+spatial_phase)/2.

    def static_center_grating(self,
                              i_screen_time=1, # >= onset for static stim
                              center=(40, 20),
                              center_radius=5,
                              center_contrast=1.,
                              center_spatial_freq = 0.1,
                              center_spatial_phase = np.pi,
                              center_theta = np.pi/6., **args):

        if i_screen_time==1:
            self.initialize_static()
        
        x_theta = (self.SCREEN['x_2d']-center[0]) * np.cos(center_theta) + (self.SCREEN['y_2d']-center[1]) * np.sin(center_theta)
        y_theta = -(self.SCREEN['x_2d']-center[0]) * np.sin(center_theta) + (self.SCREEN['y_2d']-center[1]) * np.cos(center_theta)

        cond = ((x_theta**2+y_theta**2)<center_radius**2)
        self.full_array[i_screen_time,:,:][cond] = 0.5-center_contrast*np.cos(2*np.pi*center_spatial_freq*x_theta[cond]+center_spatial_phase)/2.

    def static_surround_grating(self,
                                i_screen_time=1, # >= onset for static stim
                                center=(40, 20),
                                center_radius=5,
                                surround_radius=10,
                                surround_contrast=1.,
                                surround_spatial_freq = 0.1,
                                surround_spatial_phase = np.pi,
                                surround_theta = np.pi/6., **args):

        if i_screen_time==1:
            self.initialize_static()
        
        x_theta = (self.SCREEN['x_2d']-center[0]) * np.cos(surround_theta) + (self.SCREEN['y_2d']-center[1]) * np.sin(surround_theta)
        y_theta = -(self.SCREEN['x_2d']-center[0]) * np.sin(surround_theta) + (self.SCREEN['y_2d']-center[1]) * np.cos(surround_theta)

        cond = ((x_theta**2+y_theta**2)>=center_radius**2) & ((x_theta**2+y_theta**2)<=surround_radius**2)
        self.full_array[i_screen_time,:,:][cond] = 0.5-surround_contrast*np.cos(2*np.pi*surround_spatial_freq*x_theta[cond]+surround_spatial_phase)/2.

    def static_center_surround_grating(self,
                                       i_screen_time=1, # >= onset for static stim
                                       center=(40, 20),
                                       center_radius=5,
                                       center_contrast=1.,
                                       center_theta = np.pi/6.,
                                       center_spatial_freq = 0.07,
                                       center_spatial_phase = np.pi,
                                       surround_contrast=1.,
                                       surround_radius=20,
                                       surround_spatial_freq = 0.07,
                                       surround_spatial_phase = np.pi,
                                       surround_theta = np.pi/2., **args):

        if i_screen_time==1:
            self.initialize_static()
        
        x_theta1 = (self.SCREEN['x_2d']-center[0]) * np.cos(center_theta) + (self.SCREEN['y_2d']-center[1]) * np.sin(center_theta)
        y_theta1 = -(self.SCREEN['x_2d']-center[0]) * np.sin(center_theta) + (self.SCREEN['y_2d']-center[1]) * np.cos(center_theta)

        cond1 = ((x_theta1**2+y_theta1**2)<=center_radius**2)
        self.full_array[i_screen_time,:,:][cond1] = 0.5-center_contrast*np.cos(2*np.pi*center_spatial_freq*x_theta1[cond1]+center_spatial_phase)/2.

        x_theta2 = (self.SCREEN['x_2d']-center[0]) * np.cos(surround_theta) + (self.SCREEN['y_2d']-center[1]) * np.sin(surround_theta)
        y_theta2 = -(self.SCREEN['x_2d']-center[0]) * np.sin(surround_theta) + (self.SCREEN['y_2d']-center[1]) * np.cos(surround_theta)

        cond2 = ((x_theta2**2+y_theta2**2)>=center_radius**2) & ((x_theta2**2+y_theta2**2)<=surround_radius**2)
        
        self.full_array[i_screen_time,:,:][cond2] = 0.5-surround_contrast*np.cos(2*np.pi*surround_spatial_freq*x_theta2[cond2]+surround_spatial_phase)/2.

        
    def natural_images(self,
                       i_screen_time=1,
                       image_number=3, **args):
        
        if i_screen_time==1:
            self.initialize_static()
        stim_params = self.stimulus_params
       
        filename = os.listdir(stim_params['NI_directory'])[image_number]
        img = load(os.path.join(stim_params['NI_directory'], filename))

        rescaled_img = adapt_to_screen_resolution(img, self.SCREEN)
        rescaled_img = img_after_hist_normalization(rescaled_img)
        self.full_array[i_screen_time,:,:] = rescaled_img

        
    def grey_screen(self,
                    i_screen_time=1, **args):
        self.initialize_static()
        self.full_array[i_screen_time,:,:] = 0.*self.SCREEN['x_2d']+0.5

    def black_screen(self,
                    i_screen_time=1, **args):
        self.initialize_static()
        self.full_array[i_screen_time,:,:] = 0.*self.SCREEN['x_2d']

    def white_screen(self,
                    i_screen_time=1, **args):
        self.initialize_static()
        self.full_array[i_screen_time,:,:] = 0.*self.SCREEN['x_2d']+1.

    # ------------------
    # dynamic stimuli
    # ------------------
    
    def drifting_grating(self,
                         drifting_speed_cycle_per_second = 2.,
                         spatial_freq = 0.07,
                         spatial_phase = np.pi,
                         contrast=1.,
                         theta = np.pi/6., **args):

        
        self.initialize_dynamic()
        
        x_theta = self.SCREEN['x_2d'] * np.cos(theta) + self.SCREEN['y_2d'] * np.sin(theta)
        y_theta = -self.SCREEN['x_2d']* np.sin(theta) + self.SCREEN['y_2d'] * np.cos(theta)

        for it, t in enumerate(self.screen_time_axis_from_t0):
            
            self.full_array[it+self.it0,:,:] = 0.5-contrast*np.cos(2*np.pi*spatial_freq*x_theta+spatial_phase+2.*np.pi*drifting_speed_cycle_per_second*t)/2.


    def sparse_noise(self, seed=0,
                     SN_square_size=4.,
                     SN_sparseness=0.1,
                     SN_noise_mean_refresh_time=0.5,
                     SN_noise_rdm_jitter_refresh_time=0.2,
                     **args):

        self.initialize_dynamic()
        np.random.seed(seed)
                                   
        Nx = np.floor(self.SCREEN['width']/SN_square_size)+1
        Ny = np.floor(self.SCREEN['height']/SN_square_size)+1

        Ntot_square = Nx*Ny
        
        # an estimate of the number of shifts needed
        nshift = int((self.tstop-self.t0)/SN_noise_mean_refresh_time)+10
        events = np.cumsum(np.abs(SN_noise_mean_refresh_time+\
                                  np.random.randn(nshift)*SN_noise_rdm_jitter_refresh_time))
        events = np.concatenate([[self.t0], self.t0+events[events<self.tstop], [self.tstop]]) # restrict to stim
        x, y = self.SCREEN['x_2d'], self.SCREEN['y_2d']
        for t1, t2 in zip(events[:-1], events[1:]):
            
            Loc = np.random.choice(np.arange(Ntot_square), int(SN_sparseness*Ntot_square), replace=False)
            Val = np.random.choice([0, 1], int(SN_sparseness*Ntot_square))

            Z = 0.5+0.*x
            
            for r, v in zip(Loc, Val):
                x0, y0 = (r % Nx)*SN_square_size, int(r / Nx)*SN_square_size
                cond = (x>=x0) & (x<x0+SN_square_size) & (y>=y0) & (y<y0+SN_square_size)
                Z[cond] = v

            it1 = self.from_time_to_array_index(t1)
            it2 = self.from_time_to_array_index(t2)

            self.full_array[it1:it2,:,:] = Z


    def dense_noise(self, seed=0,
                    DN_square_size=4.,
                    DN_noise_mean_refresh_time=0.5,
                    DN_noise_rdm_jitter_refresh_time=0.2,
                    **args):

        np.random.seed(seed)
        self.initialize_dynamic()
                                   
        Nx = np.floor(self.SCREEN['width']/DN_square_size)+1
        Ny = np.floor(self.SCREEN['height']/DN_square_size)+1

        Ntot_square = Nx*Ny
        nshift = int((self.tstop-self.t0)/DN_noise_mean_refresh_time)+10
        events = np.cumsum(np.abs(DN_noise_mean_refresh_time+\
                                  np.random.randn(nshift)*DN_noise_rdm_jitter_refresh_time))
        events = np.concatenate([[self.t0], self.t0+events[events<self.tstop], [self.tstop]]) # restrict to stim

        x, y = self.SCREEN['x_2d'], self.SCREEN['y_2d']
        for t1, t2 in zip(events[:-1], events[1:]):
            
            Loc = np.arange(int(Ntot_square))
            Val = np.random.choice([0, 1], int(Ntot_square))

            Z = 0.5+0.*x
            
            for r, v in zip(Loc, Val):
                x0, y0 = (r % Nx)*DN_square_size, int(r / Nx)*DN_square_size
                cond = (x>=x0) & (x<x0+DN_square_size) & (y>=y0) & (y<y0+DN_square_size)
                Z[cond] = v

            it1 = self.from_time_to_array_index(t1)
            it2 = self.from_time_to_array_index(t2)

            self.full_array[it1:it2,:,:] = Z
            
    # def gaussian_blob_static(self,
    #                          i_screen_time=1,
    #                          seed=0, **args):

    #     self.initialize_static()
    #     stim_params = self.stimulus_params

    #     spatial = gaussian_2d(self.SCREEN['x_2d'], self.SCREEN['y_2d'],
    #                           mu=stim_params['blob_center'],
    #                           sigma=stim_params['blob_size'])
    #     spatial *= stim_params['blob_amplitude']/spatial.max()


    #     self.full_array[i_screen_time,:,:] = spatial

        
    def gaussian_blob_appearance(self,
                                 blob_center=[25.,15.],
                                 blob_size=[2.,2.],
                                 blob_amplitude=1.,
                                 blob_rise_time=1.,
                                 blob_time=2.5,
                                 **args):
        
        self.initialize_dynamic()
        self.full_array *= 0 # to force black screen before
        

        spatial = gaussian_2d(self.SCREEN['x_2d'], self.SCREEN['y_2d'],
                              mu=blob_center, sigma=blob_size)
        spatial *= blob_amplitude/spatial.max()
        temporal = np.exp(-(self.screen_time_axis_from_t0-blob_time)**2/\
                          blob_rise_time**2)
        temporal /= temporal.max()

        for i, t in enumerate(self.screen_time_axis_from_t0):
            self.full_array[i+self.it0,:,:] = temporal[i]*spatial

    
    def center_surround_protocols(self,
                                  center_delay=1.,
                                  center_duration=2.,
                                  surround_delay=0.,
                                  surround_duration=2.,
                                  **args):

        self.initialize_dynamic()

        for i, t in enumerate(self.screen_time_axis_from_t0):
            if (t>=surround_delay) and (t<=surround_delay+surround_duration) and (t>=center_delay) and (t<=center_delay+center_duration):
                self.static_center_surround_grating(i+self.it0, **args)
            elif (t>=surround_delay) and (t<=surround_delay+surround_duration):
                self.static_surround_grating(i+self.it0, **args)
            elif (t>=center_delay) and (t<=center_delay+center_duration):
                self.static_center_surround_grating(i+self.it0, **args)
                

        

if __name__=='__main__':

    
    from plots import plot
    
    stim = visual_stimulus(sys.argv[-1])
    
    stim_plot= plot(stimulus=stim, graph_env_key='visual_stim')
    
    stim_plot.screen_movie(stim)
    # if stim.stimulus_params['static']:
    #     stim_plot.screen_plot(stim.full_array[0,:,:])
    # else:
    #     stim_plot.screen_movie(stim)
    
    stim_plot.show()
    


    
