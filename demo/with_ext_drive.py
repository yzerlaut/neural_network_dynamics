import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pylab as plt
import main as ntwk

dt, tstop = 0.1, 100.
t_array = ntwk.arange(int(tstop/dt))*dt

################################################################
## ------ Construct populations with their equations -------- ##
## ------------- with recurrent connections ----------------- ##
################################################################

NEURONS = [{'name':'Exc', 'N':4000, 'type':'LIF2'},
           {'name':'Inh', 'N':1000, 'type':'LIF2_Vthre_-55'}]

M = ntwk.init_syn_and_conn_matrix(NEURONS, pconn=0.02)
M[0,0]['Q'], M[0,1]['Q'] = .1, .1 # almost no recurrent excitation
M[1,0]['Q'], M[1,1]['Q'] = 10, 10

NTWK = ntwk.build_populations(NEURONS, M,
                              with_raster=True, with_Vm=4)

ntwk.build_up_recurrent_connections(NTWK)

#######################################
########### AFFERENT INPUTS ###########
#######################################

EAff = {'Q':7., 'N':400, 'pconn':0.1}
faff = 1.
# # afferent excitation onto cortical excitation and inhibition
for i, cond in zip(range(2), ['ExcExc', 'ExcInh']): # both on excitation and inhibition
    ntwk.construct_feedforward_input(NTWK, NTWK['POPS'][i], EAff, t_array, faff+0.*t_array,
                                     conductanceID=cond,
                                     with_presynaptic_spikes=True,
                                     SEED=int(37*faff+i)%37)


################################################################
## --------------- Initial Condition ------------------------ ##
################################################################
ntwk.initialize_to_rest(NTWK)


#####################
## ----- Run ----- ##
#####################
network_sim = ntwk.collect_and_run(NTWK, tstop=tstop, dt=dt)

# ######################
# ## ----- Plot ----- ##
# ######################
ii=0
for pop in NTWK['RASTER']:
    plt.plot(pop.t/ntwk.ms, ii+pop.i, 'o')
    ii+=np.array(pop.i).max()
ntwk.set_plot(plt.gca(), ['bottom'], xlabel='time (ms)', yticks=[])
ntwk.show()

for i in range(4):
    plt.plot(NTWK['VMS'][0][i].t/ntwk.ms, NTWK['VMS'][0][i].V/ntwk.mV)
ntwk.show()


