import os, sys, pathlib, json
import numpy as np
import matplotlib.pylab as plt

module_path = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(module_path)
import ntwk
from utils import plot_tools as pt
from utils import json
from matplotlib.pylab import figure, subplot2grid

configs = ['null-activity',
           'ext-driven-AI',
           'saturated-act',
           'self-sustained']

if len(sys.argv)>2:

    if sys.argv[1]=='plot':
        """
        Plot
        """
        if sys.argv[2] in configs:
            configs = [sys.argv[2]]

            fig = figure(figsize=(8, 8))
            fig.subplots_adjust(wspace=0.7, hspace=0.5, left=0, right=0.99,
                                top=0.95, bottom=0.05)
            AX, start = {}, 0
            for key, width in zip(['config', 'TF', 'MF', 'sim-full', 'sim-zoom'],
                                  [1, 3, 2, 4, 3]):
                AX['%s-width'%key] = width
                AX['%s-start'%key] = start
                start += width

            for i, config in enumerate(configs):

                for key in ['config', 'TF', 'MF', 'sim-full', 'sim-zoom']:
                    AX['%s-%i'%(key,i)] = subplot2grid((4, start),
                                        (i, AX['%s-start'%key]),
                                        colspan=AX['%s-width'%key])
            
                tf_file = os.path.join(module_path,
                                       'demo', 'from_MF_review',\
                                       'data', 'tf', '%s.npy' % config)
                tf = np.load(tf_file, allow_pickle=True).item()


                ntwk.plots.tf_2_variables(tf, ax=AX['TF-%i'%i],
                                          xkey='F_Exc', ckey='F_Inh')

            AX['config-0'].set_title('Parameters')
            AX['TF-0'].set_title('Transfer Function')
            AX['MF-0'].set_title('Mean Field Analysis')
            AX['sim-full-0'].set_title('Network Simulation')
            AX['sim-zoom-0'].set_title('$V_m$ dynamics')
            pt.show()
    
    if sys.argv[1]=='tf':
        """
        Run Transfer Function characterization
        """
        if sys.argv[2] in configs:

            config_file = os.path.join(module_path,
                                'demo', 'from_MF_review',\
                                'configs', '%s.json' % sys.argv[2])
            Model = json.load(config_file)

            tf_file = os.path.join(module_path,
                                   'demo', 'from_MF_review',\
                                   'data', 'tf', '%s.npy' % sys.argv[2])

            Model['filename'] = tf_file

            Model['NRN_KEY'] = 'Exc' # we scan this population

            if 'quick' in sys.argv:
                Model['N_SEED'] = 1
                N = 4
            else:
                Model['N_SEED'] = 3
                N = 10

            Model['POP_STIM'] = ['Exc', 'Inh']

            Model['F_Exc_array'] = np.linspace(Model['F_Exc_min'],
                                               Model['F_Exc_max'], 4*N)
            Model['F_Inh_array'] = np.linspace(Model['F_Inh_min'],
                                               Model['F_Inh_max'], N)
            
            ntwk.transfer_functions.generate(Model)

        else:
            print(' arg must be one of:', configs)


    if sys.argv[1]=='ntwk':
        """
        Run Network Simulation
        """
        if sys.argv[2] in configs:

            config_file = os.path.join(module_path,
                                'demo', 'from_MF_review',\
                                'configs', '%s.json' % sys.argv[2])
            Model = json.load(config_file)

            ## we build and run the simulation
            NTWK = ntwk.build.populations(Model, ['Exc', 'Inh'],
                                          with_raster=True,
                                          with_Vm=4,
                                          verbose=True)

            ntwk.build.recurrent_connections(NTWK, SEED=5,
                                             verbose=True)

            ntwk_file = os.path.join(module_path,
                                   'demo', 'from_MF_review',\
                                   'data', 'ntwk-%s.h5' % sys.argv[2])

            Model['filename'] = tf_file

            Model['NRN_KEY'] = 'Exc' # we scan this population

            if 'quick' in sys.argv:
                Model['N_SEED'] = 1
                N = 4
            else:
                Model['N_SEED'] = 3
                N = 10

            Model['POP_STIM'] = ['Exc', 'Inh']

            Model['F_Exc_array'] = np.linspace(Model['F_Exc_min'],
                                               Model['F_Exc_max'], 4*N)
            Model['F_Inh_array'] = np.linspace(Model['F_Inh_min'],
                                               Model['F_Inh_max'], N)
            
            ntwk.transfer_functions.generate(Model)

        else:
            print(' arg must be one of:', configs)



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



    
