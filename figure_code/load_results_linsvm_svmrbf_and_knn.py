#from standard library
import json
import urllib.request

#from Anaconda
import numpy as np

#constants
DATA_URL = 'http://www.nicholdav.info/static/results_dict.json'
BIRD_NAMES_DICT = {
    'bird 1':'gr41rd51',
    'bird 2':'gy6or6',
    'bird 3':'or60yw70',
    'bird 4':'bl26lb16'
    }
BIRD_NAME_LIST = ['bird 1','bird 2','bird 3','bird 4']
NUM_SONGS_TO_TEST = list(range(3,16,3)) + [21,27,33,39]
# i.e., [3,6,9,...15,21,27,33,39].

response = urllib.request.urlopen(DATA_URL).read()
results_dict = json.loads(response.decode('utf-8'))
