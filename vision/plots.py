import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pylab as plt
from matplotlib.patches import Circle
from datavyz.main import graph_env

from vision.gabor_filters import gabor
from vision.earlyVis_model import earlyVis_model
from vision.stimuli import visual_stimulus
from vision.virtual_eye_movement import virtual_eye_movement


class plot:

    def __init__(self,
                 model = None,
                 stimulus = None,
                 eye_movement = None,
                 graph_env_key='manuscript'):

        self.ge = graph_env(graph_env_key)

        # need to find screen parameters
        if stimulus is not None:
            self.stimulus = stimulus
            self.SCREEN = stimulus.SCREEN
        elif model is not None:
            self.model = model
            self.SCREEN = model.SCREEN
        elif eye_movement is not None:
            self.eye_movement = eye_movement
            self.SCREEN = eye_movement.SCREEN
        else:
            print('need to find screen parameters')
            
        
    def screen_plot(self, array,
                    ax=None,
                    vmin=0, vmax=1,
                    Ybar_label='',
                    figsize=(1.,1.),
                    with_scale_bar=True,
                    Xbar_label='10$^o$'):
        """
        plotting screeen display within a graph_env
        """
        if ax is None:
            fig, ax = self.ge.figure(figsize=figsize,
                                     left=0.02, bottom=0.01, top=0.3, right=.4)
        else:
            fig = None

        self.ge.matrix(array,
                         colormap=self.ge.binary_r,
                         bar_legend=None,
                         vmin=vmin, vmax=vmax,
                         ax=ax)

        if with_scale_bar:
            self.ge.draw_bar_scales(ax,
                                Xbar=10.*self.SCREEN['dpd'], Ybar_label=Ybar_label,
                                Ybar=10.*self.SCREEN['dpd'], Xbar_label=Xbar_label,
                                xyLoc=(-0.02*self.SCREEN['Xd_max'],
                                       1.02*self.SCREEN['Yd_max']),
                                loc='left-top')
        ax.axis('equal')
        ax.axis('off')
        return fig, ax

    def screen_movie(self, stim,
                     vmin=0, vmax=1,
                     subsampling=2):

        if stim.stimulus_params['static']:
            full_array = np.array([stim.get(tt) for tt in stim.screen_time_axis[::subsampling]])
        else:
            full_array = stim.full_array[::subsampling,:,:]
        
        return self.ge.movie(full_array,
                             time=stim.screen_time_axis[::subsampling],
                             cmap=self.ge.binary_r,
                             vmin=vmin, vmax=vmax,
                             aspect='equal',
                             annotation_text='t=%.2fs')
        

    def plot_RF_properties(self, nmax=100000,
                           cell_list = None):

        if cell_list is None:
            cell_list = np.arange(min([9, self.model.Ncells]))

        axes_extents=[[[1,2], [4,2], [1,2]]]

        for i4 in range(int(len(cell_list)/3)):
            axes_extents.append([[2,1], [2,1], [2,1]])
        
        fig, AX = self.ge.figure(axes_extents=axes_extents, figsize=(.6,.7),
                                 hspace=0.5, wspace=0.5, left=0.2, bottom=0.01, top=.6)
        for ax in AX[0]:
            ax.axis('off')

        acb = plt.axes([.84, .73, .02, .2])

        Z = 0*self.SCREEN['x_2d']
        for i in range(np.min([nmax, self.model.Ncells])):
            z = self.model.cell_gabor(i)
            Z += z/len(cell_list)
        self.screen_plot(Z,
                         vmin=-np.abs(Z).max(), vmax=np.abs(Z).max(),
                         ax=AX[0][1])
        self.ge.build_bar_legend_continuous(acb, self.ge.binary_r,
                                            bounds=[-np.abs(Z).max(),np.abs(Z).max()],
                                            ticks=[-np.abs(Z).max(),0,np.abs(Z).max()], 
                                            label='filters weighted sum')
        
        self.ge.annotate(AX[0][1], 'all %i cells superimposed' % self.model.Ncells, (.25,.95))

        for i, ax in zip(cell_list, np.array(AX[1:]).flatten()):
            z = self.model.cell_gabor(i)
            self.screen_plot(z,
                             vmin=-1, vmax=1,
                             ax=ax)
            self.ge.annotate(ax, 'cell %i' % (i+1), (1,0), xycoords='data', color='w')
            
        return fig

            

    def gabor_plot(self, i,
                   ax=None,
                   x0=0, y0=0,
                   factor=1.):

        x, y = np.meshgrid(np.linspace(-1, 1)*2.5*factor*self.model.CELLS['size'][i],
                           np.linspace(-1, 1)*2.5*factor*self.model.CELLS['size'][i], indexing='ij')

        gb = gabor(x, y,
                   x0=x0, y0=y0,
                   freq=self.model.CELLS['freq'][i],
                   size=self.model.CELLS['size'][i],
                   beta=self.model.CELLS['beta'][i],
                   theta=self.model.CELLS['theta'][i],
                   psi=self.model.CELLS['psi'][i])
        _, ax, _ = self.ge.matrix(gb,
                                 vmin=-np.abs(gb).max(),
                                 vmax=np.abs(gb).max(),
                                 colormap=self.ge.binary_r,
                                 ax=ax)
        ax.axis('off')


    def show_cell_props_distrib(self):

        fig, AX = self.ge.figure((1,7), figsize=(.49, .7), wspace=0.6, right=.9, left=1,
                                 top=.1, bottom=.8)

        for ax, key in zip(AX,
                           ['x0', 'y0', 'size', 'freq', 'beta', 'theta', 'psi']):
            x = np.linspace(*self.model.params['rf_'+key], 11)

            self.ge.hist(self.model.CELLS[key], bins=x, ax=ax)
            self.ge.set_plot(ax, ['bottom'], xlabel=key, xlim=[x[0], x[-1]])
            
        self.ge.set_plot(AX[0], xlabel='x0 ($^o$)', xticks=[20, 30],
                         ylabel='cell #     ', ylabelpad=-6)
        self.ge.set_plot(AX[1], ['bottom'], xlabel='y0 ($^o$)', xticks=[17, 23])
        self.ge.set_plot(AX[2], ['bottom'], xlabel='$\sigma$ ($^o$)', xticks=[3, 8])
        self.ge.set_plot(AX[3], ['bottom'], xlabel='$f$ (c./$^o$)', xticks=[0.05, 0.1],
                         xticks_labels=['0.05  ', ' 0.1'])
        self.ge.set_plot(AX[4], ['bottom'], xlabel='$\\beta$', xticks=[1, 2])
        self.ge.set_plot(AX[5], ['bottom'], xlabel='$\\theta$', xticks=[0, np.pi/2, np.pi],
                         # xticks_labels=['0', r'$\frac{\pi}{2}$', '$\pi$  '])
                         xticks_labels=['0', '$\pi$/2', '$\pi$  '])
        self.ge.set_plot(AX[6], ['bottom'], xlabel='$\\psi$', xticks=[0, np.pi, 2*np.pi],
                         xticks_labels=['  0', '$\pi$', '2$\pi$'])
        
        return fig

        
    def RF_location_plot(self, i, ax=None):
        
        _, ax, _ = self.ge.matrix(0.4+0*self.model.SCREEN['x_2d'],
                                 vmin=0., vmax=1.,
                                 colormap=self.ge.binary_r,
                                 ax=ax)
        circ=plt.Circle((self.model.CELLS['x0'][i]*self.SCREEN['dpd'],
                         self.model.CELLS['y0'][i]*self.SCREEN['dpd']),
                        radius=self.model.CELLS['size'][i]*self.SCREEN['dpd'],
                        color='w', fill=False, lw=3)
        ax.add_patch(circ)
        ax.axis('off')

    def SEM_plot(self):

        fig, AX = self.ge.figure(axes_extents=[[[1,2], [2,2], [1,2]],
                                               [[4,1]],
                                               [[4,1]]],
                                 hspace=0.2, wspace=0.3, top=0.1, left=1., bottom=1., figsize=(.6, .5))
        for ax in AX[0]:
            ax.axis('off')
        
        visual_stim = visual_stimulus('grey-screen',
                                      stimulus_params=self.model.params,
                                      screen_params=self.model.params)
        
        self.screen_plot(visual_stim.get(1), vmin=0, vmax=1, ax=AX[0][1])


        self.ge.multicolored_line(self.model.EM['x']*self.model.params['screen_dpd'],
                                  self.model.EM['y']*self.model.params['screen_dpd'],
                                  np.linspace(0, 1, len(self.model.EM['x'])),
                                  ax=AX[0][1], lw=1)
        
        self.ge.multicolored_line(self.model.t_screen, self.model.EM['x'],
                                  np.linspace(0, 1, len(self.model.EM['x'])),
                                  ax=AX[1][0], lw=2)
        AX[1][0].plot([self.model.t_screen[0], self.model.t_screen[-1]],
                      self.model.params['screen_width']/2*np.ones(2), 'k:', lw=0.5)
        self.ge.set_plot(AX[1][0], ['left'],
                         xlim=[self.model.t_screen[0], self.model.t_screen[-1]],
                         ylabel='x ($^o$)', ylim=[0, self.model.params['screen_width']])
        self.ge.multicolored_line(self.model.t_screen, self.model.EM['y'],
                                  np.linspace(0, 1, len(self.model.EM['x'])),
                                  ax=AX[2][0], lw=2)
        AX[2][0].plot([self.model.t_screen[0], self.model.t_screen[-1]],
                      self.model.params['screen_height']/2*np.ones(2), 'k:', lw=0.5)
        self.ge.set_plot(AX[2][0], xlabel='time (s)',
                         xlim=[self.model.t_screen[0], self.model.t_screen[-1]],
                         ylabel='y ($^o$)', ylim=[0, self.model.params['screen_height']])
        
        return fig, AX
        
        
    def protocol_plot(self, cell_plot=3):

        if type(cell_plot) in (list, np.ndarray, np.array):
            pass
        else:
            cell_plot = range(cell_plot)

            
        axes_extents = [[[1,3], [1,3], [1,3], [1,3], [1,3]],
                        [[5,1]],
                        [[5,3]],
                        [[5,3]]]
        for i in cell_plot:
            axes_extents.append([[5,2]])
            axes_extents.append([[5,2]])
        axes_extents.append([[5,3]])
        
        fig, AX = self.ge.figure(axes_extents=axes_extents,
                                 figsize=(.7, .2),
                                 hspace=0.8, top=0.8, left=1.2, bottom=2.)

        visual_stim = visual_stimulus(self.model.params['stimulus_key'],
                                      stimulus_params=self.model.params,
                                      screen_params=self.model.params)

        for i in range(5): # showing the screen stim at 5 successive timepoints
            it=int(self.model.t[-1]/self.model.dt/5)*i
            its=int(self.model.t_screen[-1]/self.model.dt_screen/5)*i
            self.screen_plot(visual_stim.get(self.model.t[it]),
                        Xbar_label='10$^o$  t=%.1fs' % self.model.t[it], ax=AX[0][i])
            self.ge.multicolored_line(self.model.EM['x']*self.model.params['screen_dpd'],
                                      self.model.EM['y']*self.model.params['screen_dpd'],
                                      np.linspace(0, 1, len(self.model.EM['x'])),
                                      ax=AX[0][i], lw=0.5)
            AX[0][i].scatter([self.model.EM['x'][its]*self.model.params['screen_dpd']],
                             [self.model.EM['y'][its]*self.model.params['screen_dpd']],
                             alpha=1, s=10,
                             color=self.ge.cool(np.linspace(0, 1, len(self.model.EM['x']))[its]))

        # a nice time axis with the multiline
        AX[1][0].axis('off')
        upper_plot_pos = AX[2][0].get_position()
        time_axis = plt.axes([upper_plot_pos.x0, upper_plot_pos.y1,
                              upper_plot_pos.x1-upper_plot_pos.x0,
                              .2*(upper_plot_pos.y1-upper_plot_pos.y0)])
        self.ge.multicolored_line(self.model.t_screen, np.zeros(len(self.model.t_screen)),
                                  np.linspace(0.3, 1., len(self.model.t_screen)),
                                  ax=time_axis, lw=3)
        self.ge.annotate(time_axis, 'time (s)\n', (.5, 1.), ha='center')
        self.ge.set_plot(time_axis, [],
                         xlim=[self.model.t[0], self.model.t[-1]], ylim=[-1,1])

        ixc, iyc = int(self.SCREEN['Xd_max']/2),int(self.SCREEN['Yd_max']/2)
        luminance_at_center = [visual_stim.get(t)[ixc, iyc] for t in self.model.t_screen]
        AX[2][0].plot(self.model.t_screen, luminance_at_center)
        self.ge.set_plot(AX[2][0], ['left', 'top'], xlim=[self.model.t[0], self.model.t[-1]],
                         ylabel='norm. lum.\nat center')

        AX[3][0].plot(self.model.t_screen, self.model.EM['x']-self.model.EM['x'][0]+1, color='firebrick')
        self.ge.annotate(AX[3][0], 'x(t)', (0.,4), xycoords='data', color='firebrick')
        AX[3][0].plot(self.model.t_screen, self.model.EM['y']-self.model.EM['y'][0]-1, color='olivedrab')
        self.ge.annotate(AX[3][0], 'y(t)', (0.,-4),xycoords='data',color='olivedrab', va='top')
        self.ge.set_plot(AX[3][0], ['left'], ylabel='position\neye($^{o}$)',
                         xlim=[self.model.t[0], self.model.t[-1]],
                         ylim=.5*self.SCREEN['width']*np.array([-1,1]))

        # loop over cells:
        for k, i in enumerate(cell_plot):
            cc = self.ge.colors[k+5] # cell color
            # cell annotation
            self.ge.annotate(AX[5+2*i][0], 'cell %i' % (i+1), (0.5,1.),
                             color=cc, va='bottom', bold=True, ha='center')
            # RF feature inset
            plot_pos = AX[4+2*i][0].get_position()
            dy = .9*(plot_pos.y1-plot_pos.y0)
            inset = plt.axes([plot_pos.x0, plot_pos.y0-dy/1.5, 16./9.*dy, dy])
            self.gabor_plot(i, ax=inset)
            # RF location inset
            dy = .7*(plot_pos.y1-plot_pos.y0)
            inset2 = plt.axes([plot_pos.x1-16./9.*dy, plot_pos.y0-dy/1.7, 16./9.*dy, dy])
            self.RF_location_plot(i, ax=inset2)

            AX[4+2*i][0].plot(self.model.t, self.model.RF[i,:], color=cc)
            AX[4+2*i][0].plot(self.model.t, self.model.ADAPT[i,:], ':', color=cc)
            ymax = np.max([0.5, np.round(self.model.RF[i,:].max()+.1, 1)])
            self.ge.set_plot(AX[4+2*i][0], ['left'], ylabel='a.u.',
                             ylim = [-ymax, ymax], num_yticks=3,
                             xlim=[self.model.t[0], self.model.t[-1]])
            
            ymax = np.round(self.model.RATES[i,:].max()+1, 0)
            AX[5+2*i][0].plot(self.model.t, self.model.RATES[i,:], color=cc, lw=2)
            for spk in self.model.SPIKES[i]:
                AX[5+2*i][0].plot([spk, spk], [0, ymax], color=cc, lw=0.5)
            self.ge.set_plot(AX[5+2*i][0], ['left'], ylabel='Hz',
                             ylim = [0, ymax], num_yticks=2,
                             xlim=[self.model.t[0], self.model.t[-1]])

        for i, spk in enumerate(self.model.SPIKES):
            AX[-1][0].scatter(spk, i*np.ones(len(spk)), s=3, color=self.ge.brown)
        self.ge.set_plot(AX[-1][0], ylabel='cell ID', xlabel='time (s)',
                         xlim=[self.model.t[0], self.model.t[-1]])

        return fig

    
    def show(self):
        self.ge.show()

    def savefig(self, fig, name,
                folder='docs/figs/', dpi=200):
        if not name.endswith('.png'):
            name = name+'.png'
        fig.savefig(os.path.join(folder, name), dpi=dpi)
        


if __name__=='__main__':


    name = sys.argv[-1]

