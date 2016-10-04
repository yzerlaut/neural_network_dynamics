import numpy as np
import sys
sys.path.append('../../../common_libraries/')
from bash.python_on_server import run_command

Qe = np.linspace(6, 2, 4)
qi = 67.
# qe = 3.#np.linspace(6, 2, 4)
# Qi = np.linspace(60., 100., 4)
DESIRED=25. # Hz, desired freq.

if sys.argv[-1]=='srun':
    for qi in Qi:
        run_command('sim.py --tstop 500 --desired_mean '+str(DESIRED)+' --Qi '+str(qi)+' --Qe '+str(qe)+' -f data/varying_Qi_'+str(qi)+'.npz',\
                    with_data_file='data/varying_Qi_'+str(qi)+'.npz')
elif sys.argv[-1]=='run':
    import os
    os.system('python sim.py --tstop 500 --desired_mean '+str(DESIRED)+' --Qi '+str(qi)+' --Qe '+str(qe)+' -f data/varying_Qi_'+str(qi)+'.npz')
elif sys.argv[-1]=='qt':
    sys.path.append('/Users/yzerlaut/work/common_libraries/')
    from graphs.qt_plots import *
    app = QtWidgets.QApplication(sys.argv)
    main = Window(DATA_LIST=['data/varying_Qe_'+str(qi)+'.npz' for qe in Qe], KEYS=['plot2', 'plot3'])
    main.show()
    sys.exit(app.exec_())
else:
    from graphs.ntwk_dyn_plot import RASTER_PLOT, POP_ACT_PLOT
    from sim import *
    for qe in Qe:
        data = np.load('data/varying_Qe_'+str(qe)+'.npz')
        args = data['args'].all()
        for exc_act, inh_act in zip(data['EXC_ACTS'],data['INH_ACTS']):
            if inh_act[int(100/args.DT):].std()>0:
                fig1, ax1 = POP_ACT_PLOT(data['t_array'], [exc_act, inh_act], pop_act_zoom=[-1,80])
                fig2, ax2 = plot_autocorrel(inh_act[int(100/args.DT):], args.DT, tmax=50)
                fig1.savefig('data/FIG1_varying_Qe_'+str(qe)+'.svg')
                fig2.savefig('data/FIG2_varying_Qe_'+str(qe)+'.svg')
                fig1.savefig('data/FIG1_varying_Qe_'+str(qe)+'.png')
                fig2.savefig('data/FIG2_varying_Qe_'+str(qe)+'.png')
