import os
import numpy as np
cdir = os.path.dirname(os.path.realpath(__file__))
os.chdir('../../transfer_functions/')
os.system('ls')
discret = 10
for vthre in np.linspace(-50, -54, discret):
    os.system('python run_3d_fluct_space_charact.py EIF_Vthre_'+\
              str(vthre))
    os.system('python fit_3d_fluct_firing_response.py EIF_Vthre_'+\
              str(vthre)+' --NO_PLOT')
os.chdir(cdir)

