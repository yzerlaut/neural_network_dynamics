import os, sys
import numpy as np

DISCRETIZATION = 2
Qi_over_Qe = np.logspace(np.log(1.2)/np.log(10), np.log(1.2)/np.log(10), DISCRETIZATION)
EXT_INPUT = np.linspace(0, 6, DISCRETIZATION)

ii=0
if sys.argv[-1]=='run':
    for qiqe in Qi_over_Qe:
        for ext_input in EXT_INPUT:
            ii+=1
            os.system('python script.py --Qe '+str(5./qiqe)+\
                      ' --Qi 5. --f_ext '+str(ext_input)+' --params_scan -f data_'+str(ii)+'.npz --nrec 20')
elif sys.argv[-1]=='plot':
    for qiqe in Qi_over_Qe:
        for ext_input in EXT_INPUT:
            ii+=1
            data = np.load('data_'+str(ii)+'.npz')
            print('===========================================')
            print('for Qi/Qe', qiqe, '  Fext=', ext_input)
            print(data['mean_G_exc']) # values of each individual neuron
            print(data['std_G_exc'])
            print(data['mean_G_inh'])
            print(data['std_G_inh'])
            print(data['mean_Vm'])
            print(data['std_Vm'])
else:
    print('=====================================================================')
    print('need to specify plot or run argument !!!')
    print('=====================================================================')
        
            
