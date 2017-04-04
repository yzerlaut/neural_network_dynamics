# submodules of brian2
from brian2 import *
# building the network
from ntwk_build.syn_and_connec_library import init_syn_and_conn_matrix
from ntwk_build.syn_and_connec_construct import *
# building the stimulation
from ntwk_stim.waveform_library import double_gaussian
from ntwk_stim.connect_afferent_input import construct_feedforward_input_simple
# for plotting
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from graphs.ntwk_dyn_plot import RASTER_PLOT
from graphs.my_graph import *
plt.style.use('ggplot') # a nice plotting style
