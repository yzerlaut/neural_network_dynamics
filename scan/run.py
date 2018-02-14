import multiprocessing as mp
import numpy as np
from itertools import product
import zipfile, os

def run_scan(Model, KEYS, VALUES,
             running_sim_func,
             running_sim_func_args={},
             fix_missing_only=False,
             parallelize=True, scan_seed=10):

    np.random.seed(scan_seed)
    
    MODELS = []
    if parallelize:
        PROCESSES = []
        # Define an output queue
        output = mp.Queue()
    
    zf = zipfile.ZipFile(Model['zip_filename'], mode='w')

    Model['PARAMS_SCAN'] = {'FILENAMES':[]}
    for key in KEYS:
        Model['PARAMS_SCAN'][key] = []

    def run_func(i, output):
        running_sim_func(MODELS[i], **running_sim_func_args)
            
    i=0
    for VAL in product(*VALUES):
        Model = Model.copy()
        FN = Model['data_folder']#+'sim'
        for key, val in zip(KEYS, VAL):
            Model[key] = val
            FN += '_'+key+'_'+str(val)
            Model['PARAMS_SCAN'][key].append(val)
        FN += '_'+str(np.random.randint(100000))+'.h5'
        Model['filename'] = FN
        Model['PARAMS_SCAN']['FILENAMES'].append(FN)
        MODELS.append(Model.copy())
        
        if parallelize:
            if fix_missing_only:
                if not os.path.isfile(FN): # if it doesn't exists !
                    print('running configuration ', FN)
                    PROCESSES.append(mp.Process(target=run_func, args=(i, output)))
                else:
                    print('configuration DONE: ', f)
            else:
                PROCESSES.append(mp.Process(target=run_func, args=(i, output)))
        else:
            run_func(i, 0)
        i+=1

    if parallelize:
        # Run processes
        for p in PROCESSES:
            p.start()
        # # Exit the completed processes
        for p in PROCESSES:
            p.join()

    # writing the parameters
    np.savez(Model['zip_filename'].replace('.zip', '_Model.npz'), **Model)
    zf.write(Model['zip_filename'].replace('.zip', '_Model.npz'))
    
    for i in range(len(MODELS)):
        zf.write(MODELS[i]['filename'])

    zf.close()
    
    
if __name__=='__main__':

    Model = {'data_folder': 'data/', 'SEED':0, 'x':2, 'zip_filename':'data.zip'}
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    from recording.load_and_save import write_as_hdf5, load_dict_from_hdf5
    def running_sim_func(Model, a=0):
        NTWK = {'Model':Model, 'dt':0.1, 'tstop':1, 'NEURONS':[]}
        j = 0
        while j<1e7:
            j+=1
        write_as_hdf5(NTWK, filename=Model['filename'])

    import time
    start_time = time.time()
    print('-----------------------------------')
    print(' Without parallelization')
    run_scan(Model, ['SEED', 'x'],
             [np.arange(3), np.arange(5, 8)],
             running_sim_func, running_sim_func_args={'a':3},
             parallelize=False)
    print("--- %s seconds ---" % (time.time() - start_time))        
    print('-----------------------------------')
    start_time = time.time()
    print(' With parallelization')
    run_scan(Model, ['SEED', 'x'],
             [np.arange(3), np.arange(5, 8)],
             running_sim_func, running_sim_func_args={'a':3},
             parallelize=True)
    print("--- %s seconds ---" % (time.time() - start_time))        
