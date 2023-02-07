import json

def load(filename):
    with open(filename) as f:
        text = json.load(f)
    params = {}
    for key in text:
        if ' ' not in key:
            params[key] = text[key]
    return params
