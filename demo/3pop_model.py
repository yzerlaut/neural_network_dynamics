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
           {'name':'Inh', 'N':1000, 'type':'LIF2_Vthre_-53'},
           {'name':'DsInh', 'N':200, 'type':'LIF2_Vthre_-53'}]

M = ntwk.init_syn_and_conn_matrix(NEURONS, pconn=0) # 0 connec by default, just get the shape
# then manually setting the connectivity
M[0,0]['Q'], M[0,0]['pconn'], M[0,0]['Erev'], M[0,0]['Tsyn'] = 2., 0.05, 0., 5.
M[0,1]['Q'], M[0,1]['pconn'], M[0,1]['Erev'], M[0,1]['Tsyn'] = 2., 0.05, 0., 5.
M[1,0]['Q'], M[1,0]['pconn'], M[1,0]['Erev'], M[1,0]['Tsyn'] = 10., 0.05, -80., 5.
M[1,1]['Q'], M[1,1]['pconn'], M[1,1]['Erev'], M[1,1]['Tsyn'] = 10., 0.05, -80., 5.
M[2,1]['Q'], M[2,1]['pconn'], M[2,1]['Erev'], M[2,1]['Tsyn'] = 10., 0.02, -80., 5.

NTWK = ntwk.build_populations(NEURONS, M,
                              with_raster=True, with_Vm=4)

ntwk.build_up_recurrent_connections(NTWK)

#######################################
########### AFFERENT INPUTS ###########
#######################################

EAff = {'Q':7., 'N':400, 'pconn':0.1}
faff = 1.
# # afferent excitation onto cortical excitation and inhibition
for i, cond in enumerate(['ExcExc', 'ExcInh', 'ExcDsInh']): # both on excitation and inhibition
    ntwk.construct_feedforward_input(NTWK, NTWK['POPS'][i], EAff, t_array, faff+0.*t_array,
                                     conductanceID=cond,
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

