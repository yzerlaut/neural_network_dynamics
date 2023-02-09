import os, sys, pathlib, json
import numpy as np
import matplotlib.pylab as plt

module_path = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(module_path)
import ntwk
from utils import plot_tools as pt
from utils import params
from matplotlib.pylab import figure, subplot2grid

configs = ['null-activity',
           'ext-driven-AI',
           'saturated-act',
           'self-sustained']

def calculate_mean_firing(data,
                          window=[500,1000]):
    # from raster activity data:
    tcond = (data['tRASTER_Exc']>window[0]) & (data['tRASTER_Exc']<window[1])
    return 1e3*len(data['iRASTER_Exc'][tcond])/(window[1]-window[0])/data['N_Exc']


if len(sys.argv)>2:

    if sys.argv[1]=='plot':
        """
        Plot
        """
        if sys.argv[2] in configs:
            configs = [sys.argv[2]]

        fig = figure(figsize=(7, 9))
        fig.subplots_adjust(wspace=1.2, hspace=0.7, left=0.03, right=0.99,
                            top=0.95, bottom=0.05)
        AX, start = {}, 0
        for key, width in zip(['config', 'TF', 'MF', 'raster', 'Vm'],
                              [1, 5, 2, 3, 4]):
            AX['%s-width'%key] = width
            AX['%s-start'%key] = start
            start += width

        for i, config in enumerate(configs):

            for key in ['config', 'TF', 'MF', 'raster', 'Vm']:
                AX['%s-%i'%(key,i)] = subplot2grid((4, start),
                                    (i, AX['%s-start'%key]),
                                    colspan=AX['%s-width'%key],
                                    projection='3d' if key=='TF' else None)
        
            
            AX['config-%i'%i].axis('off')
            AX['config-%i'%i].annotate('$\\nu_{ext}$=%.1f'%10, (0.5, 0),
                    ha='center', xycoords='axes fraction')
            tf_file = os.path.join(module_path,
                                   'demo', 'from_MF_review',\
                                   'data', 'tf', '%s.npy' % config)

            tf = np.load(tf_file, allow_pickle=True).item()

            ### TF plot

            tf['Model']['NRN_KEY'] = 'Exc'
            tf['Model']['COEFFS'] = ntwk.theory.fitting_tf.fit_data(tf)

            ntwk.plots.tf_2_variables_3d(tf,
                                         ax=AX['TF-%i'%i],
                                         xkey='F_Inh', ykey='F_Exc')

            ### MF plot

            x = np.linspace(0, tf['F_Exc'].max(), 200)
            RATES = {'F_Exc':x, 'F_Inh':x}
            Fout_th = ntwk.theory.tf.TF(RATES, tf['Model'],
                                        tf['Model']['NRN_KEY'])
            inset = pt.inset(AX['MF-%i'%i], [.05,.2,.8,.6])
            inset.plot(x, Fout_th, 'k--')
            inset.plot(x, x, 'k-', lw=0.5)
            cond = Fout_th<x
            if np.sum(cond)>0:
                ipred = np.arange(len(cond))[cond][0]
            else:
                ipred = 0
            inset.plot([x[ipred]], [x[ipred]], 'ko', ms=4)
            # inset.set_title('$\\nu_{MF}$=%.1fHz' %  x[10:][ipred],
                            # style='italic', fontsize=7)

            inset.set_xlabel('$\\nu_{in}$=$\\nu_{e}$=$\\nu_{i}$ (Hz)')
            inset.set_ylabel('$\\nu_{out}$ (Hz)')
            AX['MF-%i'%i].axis('off')

            ### raster plot

            ntwk_file = os.path.join(module_path,
                                   'demo', 'from_MF_review',\
                                   'data', 'ntwk', '%s.h5' % config)

            data = ntwk.recording.load_dict_from_hdf5(ntwk_file)

            raster = pt.inset(AX['raster-%i'%i], [.0,.2,1.,.8])
            AX['raster-%i'%i].axis('off')
            ntwk.plots.ntwk_plots.raster_subplot(data, raster,
                                                 ['Inh', 'Exc', 'AffExc'],
                                                 ['r', 'g', 'b'],
                                                 [0, 1000],
                        Nmax_per_pop_cond=[data['N_Inh'], data['N_Exc'], 800],
                        subsampling=20, with_annot=False)
            raster.set_xlabel('time (ms)')
            # raster.set_title('$\\nu_{sim}$=%.1fHz' %  calculate_mean_firing(data),
                            # style='italic', fontsize=7)

            ### Vms plot

            tzoom1 = [500, 700]
            ylim = raster.get_ylim()
            raster.fill_between(tzoom1,
                                [ylim[0], ylim[0]], [ylim[1], ylim[1]],
                                color='grey', lw=0, alpha=.2)
            raster.set_ylim(ylim)
                                            
            ntwk.plots.ntwk_plots.shifted_Vms_subplot(data, AX['Vm-%i'%i],
                                                 ['Inh', 'Exc'],
                                                 ['r', 'g'], tzoom1,
                                                 spike_peak=-30, Tbar=20)
             
        AX['config-0'].set_title('Parameters\n')
        AX['TF-0'].set_title('Transfer Function\n')
        AX['MF-0'].set_title('Mean Field\nAnalysis')
        AX['raster-0'].set_title('Network Simulation\n')
        AX['Vm-0'].set_title('$V_m$ dynamics\n')

        # fig.savefig(os.path.join(os.path.expanduser('~'),
                    # 'Desktop', 'fig.png'))
        
        pt.show()
    
    if sys.argv[1]=='tf':
        """
        Run Transfer Function characterization
        """
        if sys.argv[2] in configs:

            config_file = os.path.join(module_path,
                                'demo', 'from_MF_review',\
                                'configs', '%s.json' % sys.argv[2])
            Model = params.load(config_file)

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
            Model = params.load(config_file)

            Model['tstop'] = 2000

            ## we build and run the simulation
            NTWK = ntwk.build.populations(Model, ['Exc', 'Inh', 'AffExc'],
                                          AFFERENT_POPULATIONS=['BgExc'],
                                          with_raster=True,
                                          with_Vm=4,
                                          verbose=True)

            # recurrent connections
            ntwk.build.recurrent_connections(NTWK, SEED=5,
                                             verbose=True)

            # afferent input
            t_array = ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']
            # ramp function
            Faff_array = np.array([t/200 if (t<200) else 1 for t in t_array])
            if sys.argv[2]=='self-sustained':
                Faff_array *= 1 
                Faff_array[t_array>200] = 0
            else:
                Faff_array *= Model['F_BgExc']

            ntwk.stim.construct_feedforward_input(NTWK, 'AffExc', 'BgExc',
                                                  t_array, Faff_array)
            # for i, tpop in enumerate(['Exc', 'Inh']):
                # ntwk.stim.construct_feedforward_input(NTWK, tpop, 'AffExc',
                                                      # t_array, Faff_array)
            # init
            ntwk.build.initialize_to_rest(NTWK)
            # run
            network_sim = ntwk.collect_and_run(NTWK, verbose=True)
            # write
            ntwk_file = os.path.join(module_path,
                                   'demo', 'from_MF_review',\
                                   'data', 'ntwk', '%s.h5' % sys.argv[2])
            ntwk.recording.write_as_hdf5(NTWK, filename=ntwk_file)


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



    
