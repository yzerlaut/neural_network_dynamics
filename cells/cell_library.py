"""
Some configuration of neuronal properties so that we pick up
within this file
"""

def get_neuron_params(NAME, name='', number=1, SI_units=False, verbose=True):

    BASE = NAME.split('_')[0]
    VAR, VALS = NAME.split('_')[1::2], NAME.split('_')[2::2]

    if BASE=='LIF':
        params = {'name':name, 'N':number,\
                  'Gl':10., 'Cm':150.,'Trefrac':5.,\
                  'El':-65., 'Vthre':-50., 'Vreset':-65., 'delta_v':0.,\
                  'a':0., 'b': 0., 'tauw':1e9}
    elif BASE=='EIF':
        params = {'name':name, 'N':number,\
                  'Gl':10., 'Cm':200.,'Trefrac':5.,\
                  'El':-70., 'Vthre':-50., 'Vreset':-70., 'delta_v':1.,\
                  'a':0., 'b':0., 'tauw':1e9}
    elif BASE=='AdExp':
        params = {'name':name, 'N':number,\
                  'Gl':10., 'Cm':200.,'Trefrac':5.,\
                  'El':-70., 'Vthre':-50., 'Vreset':-70., 'delta_v':2.,\
                  'a':2., 'b':20., 'tauw':500.}
    else:
        raise NameError('/!\ Cell Name not recognized /!\ \n either not implemented or mis-typed !!')

    ## POSSIBILITY TO CHANGE VALUES BY A 'FANCY' STRING INPUT !!
    for var, val in zip(VAR, VALS):
        print(var, ' changed to --> ', float(val))
        params[var] = float(val)

    if SI_units:
        print('/!\ PASSING cell parameters in SI units /!\ ')
        # mV to V
        params['El'], params['Vthre'], params['Vreset'], params['delta_v'] =\
            1e-3*params['El'], 1e-3*params['Vthre'], 1e-3*params['Vreset'], 1e-3*params['delta_v']
        # ms to s
        params['Trefrac'], params['tauw'] = 1e-3*params['Trefrac'], 1e-3*params['tauw']
        # nS to S
        params['a'], params['Gl'] = 1e-9*params['a'], 1e-9*params['Gl']
        # pF to F and pA to A
        params['Cm'], params['b'] = 1e-12*params['Cm'], 1e-12*params['b']
    else:
        if verbose:
            print('/!\ cell parameters --NOT-- in SI units /!\ ')

    return params.copy()

if __name__=='__main__':

    print(__doc__)
