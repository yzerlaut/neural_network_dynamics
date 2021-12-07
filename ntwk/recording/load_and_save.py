import numpy as np
import sys, pathlib
import brian2, os

from analyz.IO import hdf5

def write_as_hdf5(NTWK, filename='data.h5',
                  KEY_NOT_TO_RECORD=[]):

    data = {'dt':NTWK['dt']*np.ones(1), 'tstop':NTWK['tstop']*np.ones(1)}

    for key, val in NTWK['Model'].items():
        data[key] = val
            
    # we write it per population
    for ii in range(len(NTWK['NEURONS'])):
        nrn = NTWK['NEURONS'][ii]
        data[str(ii)] = {'name': nrn['name'] , 'N':nrn['N']}
        data[str(ii)+'_params'] = nrn['params']
        name = NTWK['NEURONS'][ii]['name']
        
        if ('RASTER' in NTWK) and ('RASTER' not in KEY_NOT_TO_RECORD):
            data['tRASTER_'+name] = NTWK['RASTER'][ii].t/brian2.ms
            data['iRASTER_'+name] = np.array(NTWK['RASTER'][ii].i, dtype=np.int)

        if ('POP_ACT' in NTWK) and ('POP_ACT' not in KEY_NOT_TO_RECORD):
            data['POP_ACT_'+name] = NTWK['POP_ACT'][ii].rate/brian2.Hz

        if ('VMS' in NTWK) and ('VMS' not in KEY_NOT_TO_RECORD):
            data['VMS_'+name] = np.array([vv.V/brian2.mV for vv in NTWK['VMS'][ii]])

        if ('ISYNe' in NTWK) and ('ISYNe' not in KEY_NOT_TO_RECORD):
            data['ISYNe_'+name] = np.array([vv.Ie/brian2.pA for vv in NTWK['ISYNe'][ii]])
            
        if ('ISYNi' in NTWK) and ('ISYNi' not in KEY_NOT_TO_RECORD):
            data['ISYNi_'+name] = np.array([vv.Ii/brian2.pA for vv in NTWK['ISYNi'][ii]])

        if ('GSYNe' in NTWK) and ('GSYNe' not in KEY_NOT_TO_RECORD):
            data['GSYNe_'+name] = np.array([vv.Ge/brian2.nS for vv in NTWK['GSYNe'][ii]])
            
        if ('GSYNi' in NTWK) and ('GSYNi' not in KEY_NOT_TO_RECORD):
            data['GSYNi_'+name] = np.array([vv.Gi/brian2.nS for vv in NTWK['GSYNi'][ii]])

        for aff_pop in NTWK['AFFERENT_POPULATIONS']:
            rate_key = 'Rate_%s_%s' % (aff_pop, name)
            if rate_key in NTWK:
                data[rate_key] = np.array(NTWK[rate_key], dtype=float)

            
    for aff_pop in NTWK['AFFERENT_POPULATIONS']:
        try:
            # in case the aff pop spikes were explicitely written in NTWK (see visual_input.py in demo)
            data['tRASTER_'+aff_pop] = NTWK['tRASTER_'+aff_pop]
            data['iRASTER_'+aff_pop] = NTWK['iRASTER_'+aff_pop]
        except BaseException as e:
            pass

    
    if ('iRASTER_PRE_in_terms_of_Pre_Pop' in NTWK) and ('iRASTER_PRE_in_terms_of_Pre_Pop' not in KEY_NOT_TO_RECORD):
        data['iRASTER_PRE_in_terms_of_Pre_Pop'] = np.array(NTWK['iRASTER_PRE_in_terms_of_Pre_Pop'], dtype=np.int)
        data['tRASTER_PRE_in_terms_of_Pre_Pop'] = np.array(NTWK['tRASTER_PRE_in_terms_of_Pre_Pop'], dtype=np.float)

    if ('iRASTER_PRE' in NTWK) and ('iRASTER_PRE' not in KEY_NOT_TO_RECORD):
        for jj in range(len(NTWK['iRASTER_PRE'])):
            data['iRASTER_PRE'+str(jj)] = np.array(NTWK['iRASTER_PRE'][jj], dtype=np.int)
            data['tRASTER_PRE'+str(jj)] = np.array(NTWK['tRASTER_PRE'][jj]/brian2.ms, dtype=np.float)
            
    hdf5.save_dict_to_hdf5(data, filename)
    
