import os.path as path
import os
import argparse

from numpy import lib

import librosa
import numpy as np
import joblib


def get_input_files(input_dir, classes, datasets):
    assert path.exists(input_dir), 'Input directory: {} does not exist'.format(input_dir)

    X = []
    y = []
    for cls in classes:
        for dat in datasets:
            subdir = path.join(path.join(input_dir, cls), dat)
            files = os.listdir(subdir)
            for f in files:
                X.append(path.join(subdir,f))
                y.append(cls)

    return X, y


def load_input_files(files, normalize=False):
    X = []
    for f in files:
        mels = np.load(f)['arr_0']
        if normalize:
            mels = librosa.power_to_db(mels, top_db=100)
            mels -= np.mean(mels)
            mels /= (np.std(mels) + 1e-9)
        X.append(mels)
    return X

