import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import main as ntwk

dt, tstop = 0.1, 1000.
t_array = ntwk.arange(int(tstop/dt))*dt

NTWK = [{'name':'Exc', 'N':4000, 'type':'LIF'},
        {'name':'Inh', 'N':1000, 'type':'LIF'}]

M = ntwk.init_syn_and_conn_matrix(NTWK, pconn=0.02)
M[0,0]['Q'], M[0,1]['Q'] = 7, 7
M[1,0]['Q'], M[1,1]['Q'] = 67, 67

################################################################
## ------ Construct populations with their equations -------- ##
## ------------- with recurrent connections ----------------- ##
################################################################
POPS, RASTER, VMS= ntwk.build_populations(NTWK, M,
                                          with_raster=True, with_Vm=4)
SYNAPSES = ntwk.build_up_recurrent_connections(POPS, M)

################################################################
## --------------- Initial Condition ------------------------ ##
################################################################
for i in range(2): POPS[i].V = -65*ntwk.mV # Vm at rest
# then excitation
POPS[0].GExcExc = abs(20+20*ntwk.randn(POPS[0].N))*ntwk.nS
POPS[0].GInhExc = abs(50+20*ntwk.randn(POPS[0].N))*ntwk.nS
# then inhibition
POPS[1].GExcInh = abs(20+20*ntwk.randn(POPS[1].N))*ntwk.nS
POPS[1].GInhInh = abs(50+20*ntwk.randn(POPS[1].N))*ntwk.nS

#####################
## ----- Run ----- ##
#####################
network_sim = ntwk.collect_and_run([POPS, SYNAPSES, RASTER, VMS],
                                   tstop=tstop, dt=dt)
######################
## ----- Plot ----- ##
######################
ntwk.RASTER_PLOT([pop.t/ntwk.ms for pop in RASTER], [pop.i for pop in RASTER])
ntwk.show()

for i in range(4):
    ntwk.plot(VMS[0][i].t/ntwk.ms, VMS[0][i].V/ntwk.mV)
ntwk.show()


