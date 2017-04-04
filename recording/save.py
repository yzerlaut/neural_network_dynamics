
def write_as_hdf5(RASTER, ...):

    
output = {'ispikes':np.array(RASTER[0].i), 'tspikes':np.array(RASTER[0].t/brian2.ms), 'dt':str(dt), 'tstop':str(tstop)}
    
        
    if with_Vm:
        output['i_prespikes'] = np.concatenate([i*np.ones(len(presk)) for i, presk in enumerate(PRESPKS)]).flatten()
        output['t_prespikes'] = np.concatenate([presk/brian2.ms for presk in PRESPKS]).flatten()
        output['Vm'] = np.array([vv.V/brian2.mV for vv in VMS[0]])
    if with_synaptic_currents:
        output['Ie'] = np.array([vv.Ie/brian2.pA for vv in ISYNe[0]])
        output['Ii'] = np.array([vv.Ii/brian2.pA for vv in ISYNi[0]])
