import sys
import numpy as np
from datavyz.main import graph_env

import itertools, string, sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from vision.gabor_filters import gabor

ge = graph_env('manuscript')

fig, AX = ge.figure(axes=(1,6),
                    figsize=(.8,.8),
                    wspace=.3, hspace=.4, top=0.5, left=0.1, right=0.5, bottom=0.)

SCREEN = {'width':16./9.*30, # degrees
          'height':30.,
          'dpd':3.6} # dot per degree

Nx, Ny = int(SCREEN['width']*SCREEN['dpd']), int(SCREEN['height']*SCREEN['dpd']) # pixels
x, y = np.meshgrid(np.linspace(0, SCREEN['width'], Nx),
                   np.linspace(0, SCREEN['height'], Ny))

if sys.argv[-1]=='gratings':
    x0, y0 = 0, 0

    for ax, theta in zip(np.array(AX).flatten(), np.linspace(0, (len(AX)-1)*np.pi/len(AX), len(AX))):

        # Centering and Rotation
        x_theta = (x-x0) * np.cos(theta) + (y-y0) * np.sin(theta)
        y_theta = -(x-x0) * np.sin(theta) + (y-y0) * np.cos(theta)

        spatial_freq = 0.05

        Z = np.sin(2*np.pi*spatial_freq*x_theta)

        ge.twoD_plot(x.flatten(), y.flatten(), Z.flatten()+.5, vmin=0, vmax=1, colormap=ge.binary,
                     ax=ax)
        ge.draw_bar_scales(ax, Xbar=10., Ybar_label='',
                           Ybar=10.,
                           Xbar_label='10$^o$   $\\theta$='+str(int(theta*180/np.pi))+'$^o$',
                           xyLoc=(-0.02*SCREEN['width'], 1.02*SCREEN['height']),
                           loc='left-top')
        ax.axis('equal')
        ax.axis('off')
if sys.argv[-1]=='sparse-noise':

    fig, AX = ge.figure(axes=(1,6),
                    figsize=(.8,.8),
                    wspace=.3, hspace=.4, top=0.5, left=0.1, right=0.5, bottom=0.)

    square_size = 5. # degree
    sparseness = 0.05
    
    Nx, Ny = np.floor(SCREEN['width']/square_size), np.floor(SCREEN['height']/square_size)

    Ntot_square = Nx*Ny

    for ax in np.array(AX).flatten():
        Loc = np.random.choice(np.arange(Ntot_square), int(sparseness*Ntot_square), replace=False)
        Val = np.random.choice([0, 1], int(sparseness*Ntot_square))

        Z = 0.5+0.*x

        for r, v in zip(Loc, Val):
            x0, y0 = (r % Nx)*square_size, int(r / Nx)*square_size
            cond = (x>=x0) & (x<=x0+square_size) & (y>=y0) & (y<=y0+square_size)
            Z[cond] = v

        ge.twoD_plot(x.flatten(), y.flatten(), Z.flatten(), vmin=0, vmax=1, colormap=ge.binary,
                     ax=ax)
        ge.draw_bar_scales(ax, Xbar=10., Ybar_label='',
                           Ybar=10.,
                           Xbar_label='10$^o$',
                           xyLoc=(-0.02*SCREEN['width'], 1.02*SCREEN['height']),
                           loc='left-top')
        ax.axis('equal')
        ax.axis('off')
    

# ge.show()
fig.savefig('fig.png')


