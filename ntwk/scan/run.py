import multiprocessing as mp
import numpy as np
from itertools import product
import zipfile, os, pathlib, sys

def check_success(Model):
    success = True
    for filename in Model['PARAMS_SCAN']['FILENAMES']:
        if not os.path.isfile(filename): # if it doesn't exists !
            success = False
    return success

def run_scan(Model, KEYS, VALUES,
             running_sim_func,
             running_sim_func_args={},
             fix_missing_only=False,
             parallelize=True,
             Nmax_simultaneous_processes=None,
             scan_seed=10):

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
        FN = Model['data_folder']+os.path.basename(Model['zip_filename']).replace('.zip', '')
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
                    print('configuration DONE: ', FN)
            else:
                PROCESSES.append(mp.Process(target=run_func, args=(i, output)))
        else:
            run_func(i, 0)
        i+=1

    if parallelize:
        if Nmax_simultaneous_processes is None:
            Nmax_simultaneous_processes = int(mp.cpu_count())
        print('parallelizing %i processes over %i cores' % (len(PROCESSES), Nmax_simultaneous_processes))
        # Run processes
        for i in range(len(PROCESSES)//Nmax_simultaneous_processes+1):
            for p in PROCESSES[Nmax_simultaneous_processes*i:Nmax_simultaneous_processes*(i+1)]:
                p.start()
            # # Exit the completed processes
            for p in PROCESSES[Nmax_simultaneous_processes*i:Nmax_simultaneous_processes*(i+1)]:
                p.join()
            print('multiprocessing loop: %i/%i' % (i, len(PROCESSES)//Nmax_simultaneous_processes))
            print('   n=%i/%i' % (i*len(PROCESSES), len(PROCESSES)))
            
    # writing the parameters
    np.savez(Model['zip_filename'].replace('.zip', '_Model.npz'), **Model)
    zf.write(Model['zip_filename'].replace('.zip', '_Model.npz'))
    
    for i in range(len(MODELS)):
        zf.write(MODELS[i]['filename'])

    zf.close()

    return check_success(Model)

def run_scan_with_success_check(*args, **opt_args):
    run_scan(*args, **opt_args)
    i=0
    success = False
    while not success and i<10:
        if 'fix_missing_only' in opt_args:
            opt_args['fix_missing_only']=True
            success = run_scan(*args, *opt_args)
        else:
            success = run_scan(*args, fix_missing_only=True, **opt_args)
        print('\n\n'+20*'-'+' Success Check #%i: %s'%(i+1, success)+20*'-'+'\n\n')
        i+=1


    
    
if __name__=='__main__':

    Model = {'data_folder': './data/',
            'SEED':0, 'x':2,
            'zip_filename':'data.zip'}

    import sys, pathlib, time
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from utils.hdf5 import save_dict_to_hdf5

    def running_sim_func(Model, a=0):
        NTWK = {'Model':Model, 'dt':0.1, 'tstop':1, 'NEURONS':[]}
        time.sleep(a)
        save_dict_to_hdf5(NTWK, filename=Model['filename'])

    import time
    start_time = time.time()
    print('-----------------------------------')
    print(' Without parallelization')
    run_scan(Model, ['SEED', 'x'],
             [np.arange(3), np.arange(5, 8)],
             running_sim_func, running_sim_func_args={'a':0.2},
             parallelize=False)
    print("--- %s seconds ---" % (time.time() - start_time))        
    print('-----------------------------------')
    start_time = time.time()
    print(' With parallelization')
    run_scan_with_success_check(Model, ['SEED', 'x'],
                             [np.arange(4), np.arange(4)],
                             running_sim_func, running_sim_func_args={'a':5},
                             # fix_missing_only=True,
                             parallelize=True)
    print("--- %s seconds ---" % (time.time() - start_time))        
