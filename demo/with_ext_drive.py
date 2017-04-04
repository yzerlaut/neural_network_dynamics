import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import main as ntwk

dt, tstop = 0.1, 100.
t_array = ntwk.arange(int(tstop/dt))*dt

################################################################
## ------ Construct populations with their equations -------- ##
## ------------- with recurrent connections ----------------- ##
################################################################

NTWK = [{'name':'Exc', 'N':4000, 'type':'LIF2'},
        {'name':'Inh', 'N':1000, 'type':'LIF2'}]

M = ntwk.init_syn_and_conn_matrix(NTWK, pconn=0.02)
M[0,0]['Q'], M[0,1]['Q'] = .1, .1 # almost no recurrent excitation
M[1,0]['Q'], M[1,1]['Q'] = 10, 10

POPS, RASTER, VMS= ntwk.build_populations(NTWK, M,
                                          with_raster=True, with_Vm=4)
SYNAPSES = ntwk.build_up_recurrent_connections(POPS, M)

#######################################
########### AFFERENT INPUTS ###########
#######################################
AFF_SPKS, AFF_SYNAPSES = [], []

RATE_EAff_on_exc = 1.
EAff_on_exc = {'Q':7., 'N':400, 'pconn':0.1}

EAff_exc_Spikes, EAff_exc_Synapse =\
                                    ntwk.construct_feedforward_input_simple(\
                                          POPS[0], EAff_on_exc, t_array,
                                          RATE_EAff_on_exc+0.*t_array,
                                          conductanceID='ExcExc', SEED=37)
AFF_SPKS.append(EAff_exc_Spikes)
AFF_SYNAPSES.append(EAff_exc_Synapse)
 
################################################################
## --------------- Initial Condition ------------------------ ##
################################################################
for i in range(2): POPS[i].V = -65*ntwk.mV # Vm at rest


#####################
## ----- Run ----- ##
#####################
network_sim = ntwk.collect_and_run([POPS, SYNAPSES,
                                    RASTER, VMS, AFF_SPKS, AFF_SYNAPSES],
                                   tstop=tstop, dt=dt)

######################
## ----- Plot ----- ##
######################
ntwk.RASTER_PLOT([pop.t/ntwk.ms for pop in RASTER], [pop.i for pop in RASTER])
ntwk.show()

for i in range(4):
    ntwk.plot(VMS[0][i].t/ntwk.ms, VMS[0][i].V/ntwk.mV)
ntwk.show()


