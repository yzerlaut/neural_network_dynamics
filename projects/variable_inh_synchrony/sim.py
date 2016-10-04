"""
This script sets up an afferent inhomogenous Poisson process onto the populations
"""
import numpy as np

import sys
sys.path.append('../../')

def run_sim(args, return_only_exc=False):
    ### SIMULATION PARAMETERS

    brian2.defaultclock.dt = args.DT*brian2.ms
    t_array = np.arange(int(args.tstop/args.DT))*args.DT

    NTWK = [{'name':'exc', 'N':args.Ne, 'type':'LIF'},
            {'name':'inh', 'N':args.Ni, 'type':'LIF'}]
    AFFERENCE_ARRAY = [{'Q':args.Qe_thal, 'N':args.Ne, 'pconn':args.pconn},
                       {'Q':args.Qe_thal, 'N':args.Ne, 'pconn':args.pconn}]
    rate_array = np.array([args.fext_kick if tt<args.kick_length else args.fext_stat for tt in t_array])
    
    EXC_ACTS, INH_ACTS, SPK_TIMES, SPK_IDS = [], [], [], []

    for seed in range(1, args.nsim+1):

        M = get_connectivity_and_synapses_matrix('Vogels-Abbott', number=len(NTWK))
        if args.Qe!=0.:
            M[0,0]['Q'], M[0,1]['Q'] = args.Qe, args.Qe
        if args.Qi!=0.:
            M[1,0]['Q'], M[1,1]['Q'] = args.Qi, args.Qi
            
        POPS, RASTER, POP_ACT = build_populations(NTWK, M, with_raster=True, with_pop_act=True)

        initialize_to_rest(POPS, NTWK) # (fully quiescent State as initial conditions)

        AFF_SPKS, AFF_SYNAPSES = construct_feedforward_input(POPS,
                    AFFERENCE_ARRAY, t_array, rate_array, pop_for_conductance='A', SEED=seed)
        SYNAPSES = build_up_recurrent_connections(POPS, M, SEED=seed+1)

        net = brian2.Network(brian2.collect())
        # manually add the generated quantities
        net.add(POPS, SYNAPSES, RASTER, POP_ACT, AFF_SPKS, AFF_SYNAPSES) 
        net.run(args.tstop*brian2.ms)

        EXC_ACTS.append(POP_ACT[0].smooth_rate(window='flat',\
                                               width=args.smoothing*brian2.ms)/brian2.Hz)
        INH_ACTS.append(POP_ACT[1].smooth_rate(window='flat',\
                                               width=args.smoothing*brian2.ms)/brian2.Hz)

    if return_only_exc:
        return EXC_ACTS[0]
    else:
        np.savez(args.filename, args=args, EXC_ACTS=np.array(EXC_ACTS),
             INH_ACTS=np.array(INH_ACTS), NTWK=NTWK, t_array=t_array,
             rate_array=rate_array, AFFERENCE_ARRAY=AFFERENCE_ARRAY,
             plot=get_plotting_instructions(), plot1=plot1(), plot2=plot2(), plot3=plot3())


def find_given_act_level_and_run_sim(args, desired_act=20.):

    previous_tstop = args.tstop
    args.tstop = 200. # limiting to 200ms
    df = 1 # 1 Hz increment by default
    imin = int(100/args.DT)
    exc_act = run_sim(args, return_only_exc=True)
    if exc_act[imin:].mean()>desired_act:
        above=True
    else:
        above=False
    while np.abs(exc_act[imin:].mean()-desired_act)>1:
        print(exc_act[imin:].mean())
        if exc_act[imin:].mean()>desired_act:
            if not above:
                df /=2.
            args.fext_stat -= df
            above=True
            print('reducing to', args.fext_stat)
        if exc_act[imin:].mean()<desired_act:
            if above:
                df /=2.
            args.fext_stat += df
            above=False
            print('raising to', args.fext_stat)
        exc_act = run_sim(args, return_only_exc=True)
    print('mean of ', exc_act[imin:].mean(), 'Hz achieved for f_ext=', args.fext_stat)
    args.tstop = previous_tstop
    run_sim(args)
        

def plot_autocorrel(data, dt, tmax=50, with_fig=None):
    sys.path.append('../../common_libraries')
    from data_analysis.signanalysis import autocorrel
    import matplotlib.pylab as plt
    from graphs.my_graph import set_plot
    if with_fig is not None:
        fig, ax = with_fig
    else:
        fig, ax = plt.subplots(1, figsize=(4,3))
    fig.subplots_adjust(left=.2, bottom=.2)
    acf, t_shift = autocorrel(data, tmax, dt)
    ax.plot(t_shift, acf, 'k-')
    set_plot(ax, xlabel='time shift (ms)', ylabel='norm. AC func.', yticks=[0,.5,1.])
    return fig, ax
    
def get_plotting_instructions():
    return """
args = data['args'].all()
from graphs.ntwk_dyn_plot import RASTER_PLOT, POP_ACT_PLOT
from sim import *
# RASTER_PLOT([1e3*data['exc_spk'],1e3*data['inh_spk']], [data['exc_ids'],data['inh_ids']])
for exc_act, inh_act in zip(data['EXC_ACTS'],data['INH_ACTS']):
    POP_ACT_PLOT(data['t_array'], [exc_act, inh_act], pop_act_zoom=[-1,80])
    plot_autocorrel(inh_act[int(100/args.DT):], args.DT, tmax=50)
"""

# def plot1():
#     return """
# args = data['args'].all()
# from graphs.ntwk_dyn_plot import RASTER_PLOT, POP_ACT_PLOT
# RASTER_PLOT([1e3*data['EXC_SPIKES'],1e3*data['inh_spk']], [data['exc_ids'],data['inh_ids']], without_fig=True)
# """

def plot2():
    return """
args = data['args'].all()
print(args)
from graphs.ntwk_dyn_plot import POP_ACT_PLOT
for exc_act, inh_act in zip(data['EXC_ACTS'],data['INH_ACTS']):
    POP_ACT_PLOT(data['t_array'], [exc_act, inh_act], pop_act_zoom=[-1,80], with_fig=(figure, ax))
"""

def plot3():
    return """
args = data['args'].all()
print(args)
from sim import *
for exc_act, inh_act in zip(data['EXC_ACTS'],data['INH_ACTS']):
    plot_autocorrel(inh_act[int(100/args.DT):], args.DT, tmax=50, with_fig=(figure, ax))
"""

if __name__=='__main__':
    import argparse
    # First a nice documentation 
    parser=argparse.ArgumentParser(description=
     """ 
     Investigates what is the network response of a single spike 
     """
    ,formatter_class=argparse.RawTextHelpFormatter)

    # simulation parameters
    parser.add_argument("--DT",help="simulation time step (ms)",type=float, default=0.1)
    parser.add_argument("--tstop",help="simulation duration (ms)",type=float, default=200.)
    parser.add_argument("--nsim",help="number of simulations (different seeds used)", type=int, default=1)
    parser.add_argument("--smoothing",help="smoothing window (flat) of the pop. act.",type=float, default=0.25)
    # network architecture
    parser.add_argument("--Ne",help="excitatory neuron number", type=int, default=4000)
    parser.add_argument("--Ni",help="inhibitory neuron number", type=int, default=1000)
    parser.add_argument("--Qe", help="weight of exc. spike (0. means default)", type=float, default=0.)
    parser.add_argument("--Qe_thal", help="weight of exc. spike (0. means default)", type=float, default=7.)
    parser.add_argument("--Qi", help="weight of inhibitory spike (0. means default)", type=float, default=0.)
    parser.add_argument("--pconn", help="connection proba", type=float, default=0.05)
    parser.add_argument("--fext_kick",help="external drive KICK (Hz)",type=float, default=5.)
    parser.add_argument("--fext_stat",help="STATIONARY external drive (Hz)",type=float, default=0.)
    parser.add_argument("--kick_length",help="duration of external drive KICK (ms)",type=float, default=30.)
    # stimulation (single spike) properties
    parser.add_argument("--rise_time", help="time of the rise of the ramp (ms)", type=float, default=1.)
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("-u", "--update_plot", help="plot the figures", action="store_true")
    parser.add_argument("-q", "--qt_plot", help="plot the figures with the QT interface", action="store_true")
    parser.add_argument("--filename", '-f', help="filename",type=str, default='data.npz')
    args = parser.parse_args()

    if args.update_plot:
        data = dict(np.load(args.filename))
        data['plot'] = get_plotting_instructions()
        data['plot2']=plot2();data['plot3']=plot3()
        np.savez(args.filename, **data)
    elif args.qt_plot:
        sys.path.append('/Users/yzerlaut/work/common_libraries/')
        from graphs.qt_plots import *
        app = QtWidgets.QApplication(sys.argv)
        main = Window(DATA_LIST=[args.filename], KEYS=['plot2', 'plot3'])
        main.show()
        sys.exit(app.exec_())
    else:
        import brian2
        from ntwk_build.syn_and_connec_construct import build_populations,\
            build_up_recurrent_connections, initialize_to_rest
        from ntwk_build.syn_and_connec_library import get_connectivity_and_synapses_matrix
        from ntwk_stim.waveform_library import double_gaussian, ramp_rise_then_constant
        from ntwk_stim.connect_afferent_input import construct_feedforward_input
        from common_libraries.data_analysis.array_funcs import find_coincident_duplicates_in_two_arrays
        find_given_act_level_and_run_sim(args, desired_act=20.)
        # run_sim(args)
