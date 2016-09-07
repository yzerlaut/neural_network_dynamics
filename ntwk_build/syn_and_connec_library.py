"""
Some configuration of neuronal properties so that we pick up
within this file
"""
import numpy as np
import itertools, string

def get_connectivity_and_synapses_matrix(NAME, number=2, SI_units=False):


    # creating empty arry of objects (future dictionnaries)
    M = np.empty((number, number), dtype=object)
    # default initialisation
    for i, j in itertools.product(range(number), range(number)):
        M[i, j] = {'pconn': 0., 'Q': 0., 'Tsyn': 0., 'Erev': 0.,\
                   'name':string.ascii_uppercase[i]+string.ascii_uppercase[j]}

    if NAME=='Vogels-Abbott':
        for m in M[0,:]: m['pconn']=0.02;m['Q']=7.;m['Tsyn']=5.;m['Erev']=0.
        for m in M[1,:]: m['pconn']=0.02;m['Q']=67.;m['Tsyn']=10.;m['Erev']=-80.
        
    elif NAME=='CONFIG1':
        for m in M[0,:]: m['pconn']=0.05;m['Q']=1.;m['Tsyn']=5.;m['Erev']=0.
        for m in M[1,:]: m['pconn']=0.05;m['Q']=5.;m['Tsyn']=5.;m['Erev']=-80.
        
    else:
        print('====================================================')
        print('------------ NETWORK NOT RECOGNIZED !! ---------------')
        print('====================================================')

    if SI_units:
        print('synaptic network parameters in SI units')
        for m in M.flatten():
            m['Q'] *= 1e-9
            m['Erev'] *= 1e-3
            m['Tsyn'] *= 1e-3
    else:
        print('synaptic network parameters --NOT-- in SI units')

    return M

if __name__=='__main__':

    print(__doc__)

    M = get_connectivity_and_synapses_matrix('Vogels-Abbott')
    # M = get_connectivity_and_synapses_matrix(number)

    print('excitatory synapses M[0,:]')
    print(M[0,:])
    print('inhibitory synapses M[1,:]')
    print(M[1, :])
    
