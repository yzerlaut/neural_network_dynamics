import numpy as np
import matplotlib.pylab as plt
import sys
sys.path.append('../../../')
from common_libraries.graphs.my_graph import set_plot
data = np.load(sys.argv[-1])
fig1, ax = plt.subplots(1, figsize=(4,3))
args = data['args'].all()
plt.plot(1e3*data['exc_spk'], data['exc_ids'], 'g.')
plt.plot(1e3*data['inh_spk'], data['exc_ids'].max()+data['inh_ids'], 'r.')
set_plot(ax, xlabel='time (ms)', ylabel='neuron number', ylim=[3600,4100], xlim=[750,1000])
# fig2, ax = plt.subplots(1, figsize=(4,3))
# args = data['args'].all()
# plt.plot(1e3*data['t_array'], data['exc_act'], 'g-')
# plt.plot(1e3*data['t_array'], data['inh_act'], 'r-')
# set_plot(ax, xlabel='time (ms)', ylabel='pop. act. (Hz)')


plt.show()
