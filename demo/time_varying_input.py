import sys, pathlib
import numpy as np
import matplotlib.pylab as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import ntwk
from utils import params
from utils.signal_processing import gaussian_smoothing

paramsFile = 'ntwk/configs/RS-FS_2017/params.json'
Model = params.load(paramsFile)

## UPDATE (lower the difference between exc and inh)
Model['RecExc_b'] = 35.
Model['RecInh_deltaV'] = 1.

#######################################
###### BUILD AFFERENT INPUT ###########
#######################################

Model['tstop'] = 2150
t= ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']

Faff = 1.+0*t
for t0, w, amp in zip(\
                     [550, 950, 1050, 1550, 1850],
                     [100, 70, 150, 100, 100],
                     [1, 1.5, 1, 1, 1.3, 1]):
    Faff += amp*np.exp(-(t-t0)**2/2/w**2)
Faff *= 7.

if sys.argv[-1]=='plot':
    # ######################
    # ## ----- Plot ----- ##
    # ######################

    ## load file
    data = ntwk.recording.load_dict_from_hdf5('data/time-varying-input_data.h5')

    t0=270
    # ## plot
    fig, AX = ntwk.plots.activity_plots(data,
                                        tzoom=[t0, np.inf],
                                        COLORS=['tab:green', 'tab:red'],
                                        smooth_population_activity=0.2,
                                        subsampling=20,
                                        raster_plot_args=dict(subsampling=5))


    t= ntwk.arange(int(data['tstop']/data['dt']))*data['dt']
    cond = (t>t0)

    rFaff = (Faff[cond]-Faff[cond].mean())/Faff[cond].std()

    Finh = gaussian_smoothing(data['POP_ACT_RecInh'][cond],
                              int(10/data['dt']))
    Fexc = gaussian_smoothing(data['POP_ACT_RecExc'][cond],
                              int(10/data['dt']))
    AX[-1].plot(t[cond], Finh.mean()+rFaff*Finh.std(),
                color='tab:red', lw=6, alpha=.4)
    AX[-1].plot(t[cond], Fexc.mean()+rFaff*Fexc.std(),
                color='tab:green', lw=6, alpha=.4)

    # fig.savefig('doc/RS-FS.png')
    plt.show()

else:


    Model['Farray_AffExc'] = Faff


    # import matplotlib.pylab as plt
    # plt.plot(t, Faff)
    # plt.show()

    #######################################
    ######          RUN         ###########
    #######################################

    ntwk.quick_run.simulation(Model,
                              filename='data/time-varying-input_data.h5',
                              with_Vm=0)

    print('Results of the simulation are stored as:', 'data/time-varying-input_data.h5')
    print('--> Run \"python RS-FS.py plot\" to plot the results')

