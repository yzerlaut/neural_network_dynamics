import sys, pathlib
import numpy as np
import matplotlib.pylab as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import ntwk
from utils import params

paramsFile = 'ntwk/configs/RS-FS_2017/params.json'
Model = params.load(paramsFile)


if sys.argv[-1]=='plot':
    # ######################
    # ## ----- Plot ----- ##
    # ######################

    ## load file
    data = ntwk.recording.load_dict_from_hdf5('data/RS-FS_data.h5')

    # ## plot
    fig, AX = ntwk.plots.activity_plots(data)

    for ax in AX:
        ax.fill_between([0,350],
                        np.ones(2)*ax.get_ylim()[0],
                        np.ones(2)*ax.get_ylim()[1], color='gray', alpha=.2, lw=0)
    AX[0].annotate(' transient dyn.', (0,.95), xycoords='axes fraction',
                   va='top', color='gray', fontsize=8)
    AX[0].annotate(' stim.', (1000,4), ha='center', fontsize=9)
    fig.savefig('doc/RS-FS.png')
    plt.show()

else:

    #######################################
    ###### BUILD AFFERENT INPUT ###########
    #######################################

    t_array = ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']
    Faff_array = 4.+4*np.exp(-(t_array-1000)**2/2/100**2)

    Model['Farray_AffExc'] = Faff_array


    #######################################
    ######          RUN         ###########
    #######################################

    ntwk.quick_run.simulation(Model,
                              filename='data/RS-FS_data.h5')

    print('Results of the simulation are stored as:', 'data/RS-FS_data.h5')
    print('--> Run \"python RS-FS.py plot\" to plot the results')

