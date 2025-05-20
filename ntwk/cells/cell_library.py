"""
Some configuration of neuronal properties so that we pick up
within this file
"""

def built_up_neuron_params(Model, NRN_KEY, N=1):
    params = {'name':NRN_KEY, 'N':N}
    for key, val in Model.items():
        if key.split('_')[0]==NRN_KEY:
            # catching all model parameters that start with the population name
            params[key.replace(NRN_KEY+'_', '')] = val
    return params
        
def change_units_to_SI(params):

    print('/!\ PASSING cell parameters in SI units /!\ ')

    # mV to V, ms to s
    for key in ['El', 'Vthre', 'Vreset', 'deltaV', 'Ei', 'Ee', 
                'Te', 'Ti', 'Trefrac', 'tauw', 'Ts', 'Tsyn', 'Erev']:
        if key in params:
            params[key] *= 1e-3
    # nS to S
    for key in ['a', 'Gl', 'Qe', 'Qi', 'Q_', 'Qs']:
        if key in params:
            params[key] *= 1e-9
    # pF to F and pA to A
    for key in ['Cm', 'b']:
        if key in params:
            params[key] *= 1e-12

def initialize_AdExp_parameters(name='cell X', number=1):
    return {'name':name, 'N':number,\
            'Gl':10., 'Cm':150.,'Trefrac':5.,\
            'El':-65., 'Vthre':-50., 'Vreset':-65., 'deltaV':0.,\
            'a':0., 'b': 0., 'tauw':1e9}

def fill_NRN_params_with_values(params_to_be_filled, filling_params):
    for key, values in params_to_be_filled.items():
        if key in filling_params.keys():
            params_to_be_filled[key] = filling_params[key]
    
def get_neuron_params(NAME, name='', number=1, SI_units=False, verbose=True):
    """

    """
    BASE = NAME.split('_')[0]
    VAR, VALS = NAME.split('_')[1::2], NAME.split('_')[2::2] # to change parameters

    if BASE=='LIF':
        params = {'name':name, 'N':number,\
                  'Gl':10., 'Cm':150.,'Trefrac':5.,\
                  'El':-70, 'Vthre':-50., 'Vreset':-70., 'deltaV':0.,\
                  'a':0., 'b': 0., 'tauw':1e9}
    elif BASE=='osciLIF':
        params = {'name':name, 'N':number,\
                  'Gl':10., 'Cm':150.,'Trefrac':5.,\
                  'El':-70., 'Vthre':-50., 'Vreset':-70., 'deltaV':0.,\
                  'Ioscill_freq':3., 'Ioscill_amp':10.*20.,
                  'a':0., 'b': 0., 'tauw':1e9}
    elif BASE=='LIF2':
        params = {'name':name, 'N':number,\
                  'Gl':10., 'Cm':200.,'Trefrac':5.,\
                  'El':-70., 'Vthre':-50., 'Vreset':-70., 'deltaV':0.,\
                  'a':0., 'b': 0., 'tauw':1e9}
    elif BASE=='EIF':
        params = {'name':name, 'N':number,\
                  'Gl':10., 'Cm':200.,'Trefrac':5.,\
                  'El':-70., 'Vthre':-50., 'Vreset':-70., 'deltaV':1.,\
                  'a':0., 'b':0., 'tauw':1e9}
    elif BASE=='AdExp':
        params = {'name':name, 'N':number,\
                  'Gl':10., 'Cm':200.,'Trefrac':5.,\
                  'El':-70., 'Vthre':-50., 'Vreset':-70., 'deltaV':1.,\
                  'a':2., 'b':20., 'tauw':500.}
    else:
        raise NameError('/!\ Cell Name not recognized /!\ \n either not implemented or mis-typed !!')

    ## POSSIBILITY TO CHANGE VALUES BY A 'FANCY' STRING INPUT !!
    for var, val in zip(VAR, VALS):
        print(var, ' changed to --> ', float(val))
        params[var] = float(val)

    if SI_units:
        change_units_to_SI(params)
    else:
        if verbose:
            print('/!\ cell parameters --NOT-- in SI units /!\ ')

    return params.copy()

if __name__=='__main__':

    print(__doc__)
