import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pylab as plt
plt.style.use('ggplot') # a nice plotting style
# building the network
from ntwk_build.syn_and_connec_library import init_syn_and_conn_matrix
import ntwk_build.syn_and_connec_construct as ntwk
# building the stimulation
from ntwk_stim.waveform_library import double_gaussian
from ntwk_stim.connect_afferent_input import construct_feedforward_input_simple
# for plotting
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from graphs.ntwk_dyn_plot import RASTER_PLOT
from graphs.my_graph import show

# starting from an example
NTWK = [{'name':'exc', 'N':4000, 'type':'LIF'},
        {'name':'inh', 'N':1000, 'type':'LIF'}]

# M = get_connectivity_and_synapses_matrix('Vogels-Abbott', number=len(NTWK))
M = init_syn_and_conn_matrix(NTWK, pconn=0.02)
M[0,0]['Q'], M[0,1]['Q'] = 7, 7
M[1,0]['Q'], M[1,1]['Q'] = 67, 67

RATES = [0.,1.]

# no afferent pop
        afferent_pop = {'Q':syn['Q'], 'N':syn['N'], 'pconn':syn['pconn']}
        rate_array = RATES[i]+0.*t_array
        spikes, synapse, indices, times =\
                             construct_feedforward_input_simple(\
                                                                POPS[0], afferent_pop,
                                                                t_array, rate_array,
                                                                conductanceID=syn['name']+NTWK[0]['name'],
                                                                SEED=i+SEED, with_presynaptic_spikes=True)
print(M)
POPS, RASTER = ntwk.build_populations(NTWK, M, with_raster=True, verbose=True)

ntwk.initialize_to_rest(POPS, NTWK, M) # initialized to rest

ntwk.initialize_to_random(POPS, NTWK, M, Gmean=500., Gstd=20.)

SYNAPSES = ntwk.build_up_recurrent_connections(POPS, M)

net = ntwk.collect()
net.add(POPS, SYNAPSES, RASTER) # manually add the generated quantities

net.run(40.*ntwk.ms)

RASTER_PLOT([pop.t/ntwk.ms for pop in RASTER], [pop.i for pop in RASTER])

show()


