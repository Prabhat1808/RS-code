import os.path as path
import os
import math
import shutil
import sys
import time
import argparse

import yaml


DEFAULTS = {}
DEFAULTS['preprocess'] = 'settings\\preprocess_default.yaml'
DEFAULTS['train'] = 'settings\\train_default.yaml'
DEFAULTS['dataset'] = ['ESC-50']
DEFAULTS['class'] = ['breathing','sneezing','snoring','cough']

# type = training, preprocess
def read_settings(settings_path, type):
    filepath = settings_path
    if settings_path == '':
        filepath = DEFAULTS[type]
        filepath = path.join(os.getcwd(), filepath)

    assert path.exists(filepath), 'Config file does not exist!!'
    with open(filepath) as file:
        settings = yaml.load(file, Loader=yaml.FullLoader)

    return settings
    
# type = dataset, class
def read_names(filepath, type):
    if filepath == '':
        return DEFAULTS[type]

    with open(filepath,'r') as f:
        lines = f.readlines()
        res = []
        for l in lines:
            res.append(l.strip())
    
    return res
    