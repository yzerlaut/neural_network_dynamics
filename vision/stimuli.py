import os, sys
import numpy as np

import itertools, string, sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from datavyz.main import graph_env
from datavyz.images import load
from analyz.signal_library.classical_functions import gaussian_2d

from vision.gabor_filters import gabor
from vision.earlyVis_model import params0


stim_params0 = {
    # drifting and static gratings
    'theta':np.pi/6.,
    'cycle_per_second':2.,
    'spatial_freq':0.2,
    'static':False,
    't0':0, # in seconds
    'tstop':5, # in seconds
    # sparse noise
    'noise_mean_refresh_time':0.5, # in s
    'noise_rdm_jitter_refresh_time':0.2, # in s
    'square_size':5., # in degrees
    'sparseness':0.05,
    # gaussian blob & appearance
    'blob_center':[25.,15.],
    'blob_size':[2.,2.],
    'blob_amplitude':1.,
    'blob_rise_time':1.,
    'blob_time':2.5,
    #
}

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

def screen_plot(array,
                graph_env,
                SCREEN,
                ax=None,
                Ybar_label='',
                Xbar_label='10$^o$'):
    """
    plotting screeen display within a graph_env
    """
    if ax is None:
        fig, ax = graph_env.figure()
    else:
        fig = None
        
    graph_env.matrix(array,
                     colormap=graph_env.binary_r,
                     bar_legend=None,
                     ax=ax)
    
    graph_env.draw_bar_scales(ax,
                              Xbar=10.*SCREEN['dpd'], Ybar_label=Ybar_label,
                              Ybar=10.*SCREEN['dpd'], Xbar_label=Xbar_label,
                              xyLoc=(-0.02*SCREEN['Xd_max'],
                                     1.02*SCREEN['Yd_max']),
                              loc='left-top')
    
    ax.axis('equal')
    ax.axis('off')
    return fig, ax


class visual_stimulus:

    def __init__(self,
                 stimulus_key='drifting-grating',
                 stimulus_params=None,
                 params=None,
                 graph_env_key='visual_stim'):

        if params is not None:
            self.params = params
        else:
            self.params = params0 # above params by default

        if stimulus_params is not None:
            self.stimulus_params = stimulus_params
        else:
            self.stimulus_params = stim_params0
            
        self.SCREEN = setup_screen(self.params)
        
        self.ge = graph_env(graph_env_key)

        self.initialize_screen_time_axis()
        
        if stimulus_key=='grating':
            self.static_grating()
        elif stimulus_key=='drifting-grating':
            self.drifting_grating()
        elif stimulus_key=='sparse-noise':
            self.sparse_noise()
        elif stimulus_key=='gaussian-blob':
            self.gaussian_blob_static()
        elif stimulus_key=='gaussian-blob-appearance':
            self.gaussian_blob_appearance()


    def initialize_screen_time_axis(self):
        """
        initialize time array based on refresh rate
        """
        self.t0 = self.stimulus_params['t0']
        self.tstop = self.stimulus_params['tstop']
        self.iTmax = int((self.tstop-self.t0)*self.SCREEN['refresh_rate'])+1
        self.screen_time_axis = self.t0+np.arange(self.iTmax)/self.SCREEN['refresh_rate']

    def from_time_to_array_index(self, t):
        if self.stimulus_params['static']:
            return 0
        else:
            return int((t-self.t0)*self.SCREEN['refresh_rate'])
    
    def get(self, t):
        """
        returns the 
        """
        return self.full_array[self.from_time_to_array_index(t), :, :]
    
    def screen_plot(self, array,
                    ax=None,
                    Ybar_label='',
                    Xbar_label='10$^o$'):
        return screen_plot(array,
                           self.ge,
                           self.SCREEN,
                           ax=ax)


    ###################################################################
    ################### SET OF DIFFERENT STIMULI ######################
    ###################################################################

    def initialize_dynamic(self):
        self.stimulus_params['static'] = False
        self.full_array = np.zeros((len(self.screen_time_axis), self.SCREEN['Xd_max'], self.SCREEN['Yd_max']))
                                   
    def initialize_static(self):
        self.stimulus_params['static'] = True
        self.full_array = np.zeros((1, self.SCREEN['Xd_max'], self.SCREEN['Yd_max']))
    
    def drifting_grating(self):

        self.initialize_dynamic()
        stim_params = self.stimulus_params
        
        for key in ['cycle_per_second', 'spatial_freq', 'theta']:
            if key not in stim_params:
                raise Exception
        
        x_theta = (self.SCREEN['x_2d']-self.SCREEN['x_center']) * np.cos(stim_params['theta']) + (self.SCREEN['y_2d']-self.SCREEN['y_center']) * np.sin(stim_params['theta'])
        y_theta = -(self.SCREEN['x_2d']-self.SCREEN['x_center']) * np.sin(stim_params['theta']) + (self.SCREEN['y_2d']-self.SCREEN['y_center']) * np.cos(stim_params['theta'])

        
        for it, t in enumerate(self.screen_time_axis):
            
            self.full_array[it,:,:] = np.cos(2*np.pi*stim_params['spatial_freq']*x_theta+2.*np.pi*stim_params['cycle_per_second']*t)+.5

            # plot(x, y, Z, ax, Xbar_label='10$^o$   $\\theta$='+str(int(theta*180/np.pi))+'$^o$')

    def static_grating(self):

        self.initialize_static()
        stim_params = self.stimulus_params
        
        for key in ['spatial_freq', 'theta']:
            if key not in stim_params:
                raise Exception
        
        x_theta = (self.SCREEN['x_2d']-self.SCREEN['x_center']) * np.cos(stim_params['theta']) + (self.SCREEN['y_2d']-self.SCREEN['y_center']) * np.sin(stim_params['theta'])
        y_theta = -(self.SCREEN['x_2d']-self.SCREEN['x_center']) * np.sin(stim_params['theta']) + (self.SCREEN['y_2d']-self.SCREEN['y_center']) * np.cos(stim_params['theta'])

        self.full_array[0,:,:] = np.sin(2*np.pi*stim_params['spatial_freq']*x_theta)+.5

            
    def sparse_noise(self, seed=0):

        self.initialize_dynamic()
        np.random.seed(seed)
                                   
        stim_params = self.stimulus_params
        
        for key in ['square_size', 'sparseness', 'noise_mean_refresh_time', 'noise_rdm_jitter_refresh_time']:
            if key not in stim_params:
                raise Exception

            
        Nx = np.floor(self.SCREEN['width']/stim_params['square_size'])
        Ny = np.floor(self.SCREEN['height']/stim_params['square_size'])

        Ntot_square = Nx*Ny
        nshift = int((self.tstop-self.t0)/stim_params['noise_mean_refresh_time'])+10
        events = np.cumsum(np.abs(stim_params['noise_mean_refresh_time']+\
                                  np.random.randn(nshift)*stim_params['noise_rdm_jitter_refresh_time']))
        events = np.concatenate([[self.t0], self.t0+events[events<self.tstop], [self.tstop]]) # restrict to stim

        x, y = self.SCREEN['x_2d'], self.SCREEN['y_2d']
        for t1, t2 in zip(events[:-1], events[1:]):
            
            Loc = np.random.choice(np.arange(Ntot_square), int(stim_params['sparseness']*Ntot_square), replace=False)
            Val = np.random.choice([0, 1], int(stim_params['sparseness']*Ntot_square))

            Z = 0.5+0.*x
            
            for r, v in zip(Loc, Val):
                x0, y0 = (r % Nx)*stim_params['square_size'], int(r / Nx)*stim_params['square_size']
                cond = (x>=x0) & (x<x0+stim_params['square_size']) & (y>=y0) & (y<y0+stim_params['square_size'])
                Z[cond] = v

            it1 = self.from_time_to_array_index(t1)
            it2 = self.from_time_to_array_index(t2)

            self.full_array[it1:it2,:,:] = Z

    def gaussian_blob_static(self, seed=0):

        self.initialize_static()
        stim_params = self.stimulus_params

        spatial = gaussian_2d(self.SCREEN['x_2d'], self.SCREEN['y_2d'],
                              mu=stim_params['blob_center'],
                              sigma=stim_params['blob_size'])
        spatial *= stim_params['blob_amplitude']/spatial.max()


        self.full_array[0,:,:] = spatial

        
    def gaussian_blob_appearance(self):
        
        self.initialize_dynamic()
        stim_params = self.stimulus_params

        spatial = gaussian_2d(self.SCREEN['x_2d'], self.SCREEN['y_2d'],
                              mu=stim_params['blob_center'],
                              sigma=stim_params['blob_size'])
        spatial *= stim_params['blob_amplitude']/spatial.max()
        temporal = np.exp(-(self.screen_time_axis-stim_params['blob_time'])**2/\
                          stim_params['blob_rise_time']**2)
        temporal /= temporal.max()

        for i, t in enumerate(self.screen_time_axis):
            self.full_array[i,:,:] = temporal[i]*spatial

        print(self.full_array.max())    
    
    def natural_images(self, image_number=3):
        
        self.initialize_static()
        

    #     for i, ax in enumerate(np.array(AX).flatten()):

    #         plot(x, y, z/z.max(), ax)


    # if sys.argv[-1]=='natural-images':

    #     DIR = '/home/yann/Pictures/Imagenet/'
    #     files = os.listdir(DIR)
    #     for i, ax in enumerate(np.array(AX).flatten()):

    #         img = load(os.path.join(DIR, files[i]))
    #         flat = np.array(1000*img.flatten(), dtype=int)

    #         cumsum = np.cumsum(np.histogram(flat, bins=np.arange(1001))[0])

    #         norm_cs = np.concatenate([(cumsum-cumsum.min())/(cumsum.max()-cumsum.min())*1000, [1]])
    #         new_img = np.array([norm_cs[f]/1000. for f in flat])

    #         ge.image(new_img.reshape(img.shape), ax=ax)

    # if sys.argv[-1]=='natural-images-sem':

    #     DIR = '/home/yann/Pictures/Imagenet/'
    #     files = os.listdir(DIR)

    #     for i, ax in enumerate(np.array(AX).flatten()):

    #         img = load(os.path.join(DIR, files[i]))
    #         flat = np.array(1000*img.flatten(), dtype=int)

    #         cumsum = np.cumsum(np.histogram(flat, bins=np.arange(1001))[0])

    #         norm_cs = np.concatenate([(cumsum-cumsum.min())/(cumsum.max()-cumsum.min())*1000, [1]])
    #         new_img = np.array([norm_cs[f]/1000. for f in flat])

    #         ge.image(new_img.reshape(img.shape), ax=ax)

    #     Npoints = 10
    #     RDM_traj = [img.shape[0]*np.random.uniform(0.15, 0.85, size=Npoints),
    #                 img.shape[1]*np.random.uniform(0.15, 0.85, size=Npoints)]
    #     print(RDM_traj)
    #     for i, ax in enumerate(np.array(AX).flatten()):
    #         print(ax.get_xlim(), ax.get_ylim())
    #         ax.plot(RDM_traj[1], RDM_traj[0], lw=1, color=ge.red)

    # ge.show()
    # fig.savefig('fig.png')


    

if __name__=='__main__':

    
    stim = visual_stimulus(sys.argv[-1])

    if stim.stimulus_params['static']:
        stim.screen_plot(stim.full_array[0,:,:])
    else:
        stim.ge.movie(stim.full_array[::10,:,:],
                      time=stim.screen_time_axis[::10],
                      cmap=stim.ge.binary_r,
                      vmin=0, vmax=1,
                      annotation_text='t=%.2fs')
    
    stim.ge.show()
    


    
