import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pylab as plt
import main as ntwk

dt, tstop = 0.1, 100.
t_array = np.arange(int(tstop/dt))*dt


################################################################
## ------ Construct populations with their equations -------- ##
## ------------- with recurrent connections ----------------- ##
################################################################

NEURONS = [{'name':'Exc', 'N':4000, 'type':'LIF'},
           {'name':'Inh', 'N':1000, 'type':'LIF'}]

M = ntwk.init_syn_and_conn_matrix(NEURONS, pconn=0.02)
M[0,0]['Q'], M[0,1]['Q'] = 6, 6
M[1,0]['Q'], M[1,1]['Q'] = 67, 67

NTWK = ntwk.build_populations(NEURONS, M,
                              with_raster=True, with_Vm=4,
                              # with_synaptic_currents=True,
                              # with_synaptic_conductances=True,
                              verbose=True)
ntwk.build_up_recurrent_connections(NTWK, SEED=10)

################################################################
## --------------- Initial Condition ------------------------ ##
################################################################
for i in range(2): NTWK['POPS'][i].V = -65*ntwk.mV # Vm at rest
# then excitation
NTWK['POPS'][0].GExcExc = abs(20+20*np.random.randn(NTWK['POPS'][0].N))*ntwk.nS
NTWK['POPS'][0].GInhExc = abs(50+20*np.random.randn(NTWK['POPS'][0].N))*ntwk.nS
# # then inhibition
NTWK['POPS'][1].GExcInh = abs(20+20*np.random.randn(NTWK['POPS'][1].N))*ntwk.nS
NTWK['POPS'][1].GInhInh = abs(50+20*np.random.randn(NTWK['POPS'][1].N))*ntwk.nS

# #####################
# ## ----- Run ----- ##
# #####################
network_sim = ntwk.collect_and_run(NTWK, tstop=tstop, dt=dt)

# ######################
# ## ----- Plot ----- ##
# ######################
ii=0
for pop in NTWK['RASTER']:
    plt.plot(pop.t/ntwk.ms, ii+pop.i)
    ii+=np.array(pop.i).max()
ntwk.set_plot(plt.gca(), ['bottom'], xlabel='time (ms)', yticks=[])
ntwk.show()

for i in range(4):
    plt.plot(NTWK['VMS'][0][i].t/ntwk.ms, NTWK['VMS'][0][i].V/ntwk.mV)
ntwk.show()


