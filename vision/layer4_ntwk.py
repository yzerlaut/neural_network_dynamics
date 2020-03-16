import numpy as np
from datavyz.main import graph_env

import itertools, string, sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from vision.gabor_filters import gabor

N=5 # picking 10 cells

# ge = graph_env('visual_stim')
ge = graph_env()

fig, ax = ge.figure()

SCREEN = {'width':16./9.*8, # inches
          'height':8.,
          'dpi':20}

Xmax, Ymax = int(SCREEN['width']*SCREEN['dpi']), int(SCREEN['height']*SCREEN['dpi'])
x, y = np.meshgrid(1.0*np.arange(Xmax), 1.0*np.arange(Ymax))

RANGES = {'x0':[0.1*Xmax, 0.9*Xmax],
          'y0':[0.1*Ymax, 0.9*Ymax],
          'freq':[1./(0.05*Ymax), 1./(0.1*Ymax)],
          'size':[0.02*Ymax, 0.1*Ymax],
          'beta':[1., 2.],
          'theta':[0., np.pi/2.],
          'psi':[0., np.pi]}

CELLS = {}
for key in RANGES:
    CELLS[key] = np.random.uniform(RANGES[key][0], RANGES[key][1], size=N)

print(CELLS)    
Z = 0*x

for i in range(N):
    
    z = gabor(x, y,
              x0=CELLS['x0'][i],
              y0=CELLS['y0'][i],
              freq=CELLS['freq'][i],
              size=CELLS['size'][i],
              beta=CELLS['beta'][i],
              theta=CELLS['theta'][i],
              psi=CELLS['psi'][i])+0.5
    Z += z/N
    
ge.twoD_plot(x.flatten(), y.flatten(), Z.flatten(), vmin=0, vmax=1, colormap=ge.binary,
             ax=ax)

ge.show()
fig.savefig('fig.png')


