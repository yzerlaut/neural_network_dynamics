import os, sys, pathlib, json
import numpy as np
import matplotlib.pylab as plt

module_path = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(module_path)
import ntwk
from utils import plot_tools as pt
from matplotlib.pylab import figure, subplot2grid

configs = ['null-activity',
           'ext-driven-AI',
           'saturated-act',
           'self-sustained']

if len(sys.argv)==3:

    if sys.argv[1]=='plot':

        if sys.argv[2]=='all':

            fig = figure(figsize=(8, 8))
            fig.subplots_adjust(wspace=0.7, hspace=0.5, left=0, right=0.99,
                                top=0.95, bottom=0.05)
            AX, start = {}, 0
            for key, width in zip(['config', 'TF', 'MF', 'sim-full', 'sim-zoom'],
                                  [1, 3, 2, 4, 3]):
                AX['%s-width'%key] = width
                AX['%s-start'%key] = start
                start += width
            print(AX)
            for i, config in enumerate(configs):
                config_file = os.path.join(module_path,
                                    'demo', 'from_MF_review',\
                                    'configs', '%s.json' % config)
                for key in ['config', 'TF', 'MF', 'sim-full', 'sim-zoom']:
                    AX['%s-%i'%(key,i)] = subplot2grid((4, start),
                                        (i, AX['%s-start'%key]),
                                        colspan=AX['%s-width'%key])
                AX['config-0'].set_title('Network\nSetting')
                AX['TF-0'].set_title('Transfer Function')
                AX['MF-0'].set_title('Mean Field Analysis')
                AX['sim-full-0'].set_title('Network Simulation')
                AX['sim-zoom-0'].set_title('$V_m$ dynamics')
            

            pt.show()
    
    # ######################
    # ## ----- Plot ----- ##
    # ######################
    # data = np.load('tf_data.npy', allow_pickle=True).item()
    # ntwk.plots.tf_2_variables(data,
                              # xkey='F_Exc', ckey='F_Inh')
    # ntwk.plots.tf_2_variables(data,
    #                           xkey='F_Exc', ckey='F_Inh',
    #                           ylim=[1e-1, 100],
    #                           yticks=[0.01, 0.1, 1, 10],
    #                           yticks_labels=['0.01', '0.1', '1', '10'],
    #                           ylabel='$\\nu_{out}$ (Hz)',
    #                           xticks=[0.1, 1, 10],
    #                           xticks_labels=['0.1', '1', '10'],
    #                           xlabel='$\\nu_{e}$ (Hz)')
    # ntwk.show()
    
else:
    print("""

    run the following script with arguments to 

    python fixed_point_analysis.py tf-stim null-activity
    python fixed_point_analysis.py tf-stim null-activity

    """)

    Model['filename'] = 'tf_data.npy'
    Model['NRN_KEY'] = 'Exc' # we scan this population
    Model['tstop'] = 10000
    Model['N_SEED'] = 1 # seed repetition
    Model['POP_STIM'] = ['Exc', 'Inh']
    Model['F_Exc_array'] = np.logspace(-1, 2, 20)
    Model['F_Inh_array'] = np.logspace(-1, 2, 10)
    ntwk.transfer_functions.generate(Model)
    print('Results of the simulation are stored as:', 'tf_data.npy')
    # print('--> Run \"python 3pop_model.py plot\" to plot the results')



    
