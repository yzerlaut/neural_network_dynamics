import json, sys

def load(filename):
    with open(filename) as f:
        text = json.load(f)
    params = {}
    for key in text:
        if ' ' not in key:
            params[key] = text[key]
    return params

if __name__=='__main__':
   
    if len(sys.argv)>1:
        paramsFile = sys.argv[-1]
    else:
        paramsFile = 'ntwk/configs/The_Spectrum_of_Asynch_Dyn_2018/params.json'

    params = load(paramsFile)

    import pprint
    pprint.pprint(params)
