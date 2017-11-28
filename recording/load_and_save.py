import brian2
import numpy as np
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from recordings.hdf5 import save_dict_to_hdf5, make_writable_dict

def write_as_hdf5(NTWK, filename='data.h5'):

    data = {'dt':NTWK['dt']*np.ones(1), 'tstop':NTWK['tstop']*np.ones(1)}

    for key, val in NTWK['Model'].items():
        if (type(val)==int) or (type(val)==float):
            data[key] = np.ones(1)*val
            
    # we write it per population
    for ii in range(len(NTWK['NEURONS'])):
        nrn = NTWK['NEURONS'][ii]
        data[str(ii)] = make_writable_dict({'name': nrn['name'] , 'N':nrn['N']})
        data[str(ii)+'_params'] = make_writable_dict(nrn['params'])
        name = NTWK['NEURONS'][ii]['name']
        
        if 'RASTER' in NTWK:
            data['tRASTER_'+name] = NTWK['RASTER'][ii].t/brian2.ms
            data['iRASTER_'+name] = np.array(NTWK['RASTER'][ii].i, dtype=np.int)

        if 'POP_ACT' in NTWK:
            data['POP_ACT_'+name] = NTWK['POP_ACT'][ii].rate/brian2.Hz

        if 'VMS' in NTWK:
            data['VMS_'+name] = np.array([vv.V/brian2.mV for vv in NTWK['VMS'][ii]])

        if 'ISYNe' in NTWK:
            data['ISYNe_'+name] = np.array([vv.Ie/brian2.pA for vv in NTWK['ISYNe'][ii]])
            
        if 'ISYNi' in NTWK:
            data['ISYNi_'+name] = np.array([vv.Ii/brian2.pA for vv in NTWK['ISYNi'][ii]])

        if 'GSYNe' in NTWK:
            data['GSYNe_'+name] = np.array([vv.Ge/brian2.nS for vv in NTWK['GSYNe'][ii]])
            
        if 'GSYNi' in NTWK:
            data['GSYNi_'+name] = np.array([vv.Gi/brian2.nS for vv in NTWK['GSYNi'][ii]])

    if 'iRASTER_PRE_in_terms_of_Pre_Pop' in NTWK:
        data['iRASTER_PRE_in_terms_of_Pre_Pop'] = np.array(NTWK['iRASTER_PRE_in_terms_of_Pre_Pop'], dtype=np.int)
        data['tRASTER_PRE_in_terms_of_Pre_Pop'] = np.array(NTWK['tRASTER_PRE_in_terms_of_Pre_Pop'], dtype=np.float)

    if 'iRASTER_PRE' in NTWK:
        for jj in range(len(NTWK['iRASTER_PRE'])):
            data['iRASTER_PRE'+str(jj)] = np.array(NTWK['iRASTER_PRE'][jj], dtype=np.int)
            data['tRASTER_PRE'+str(jj)] = np.array(NTWK['tRASTER_PRE'][jj]/brian2.ms, dtype=np.float)
            
    save_dict_to_hdf5(data, filename)
    
