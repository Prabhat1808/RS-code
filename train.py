import os.path as path
import os
import argparse

import librosa
import numpy as np
import joblib
from sklearn.utils import validation
import read_config as Reader
import data_handler as Data
from models import sbcnn
from sklearn.preprocessing import LabelBinarizer
# Fix the below 2 imports according to your environment
# import tensorflow as tf
import keras
def train_model(X, y):
    model = sbcnn.build_model(frames=430, bands=40, num_labels=4)
    print (model.summary())
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    model.fit(
        x=X,
        y=y,
        batch_size=32,
        validation_split=0.3,
        epochs=100,
        shuffle=True,
    )
    


def get_arguments():
    parser = argparse.ArgumentParser()
    a = parser.add_argument

    a('--input_dir', type=str, default='', help='Input Directory containing the training files!')
    a('--classes', type=str, default='', help='Filename containing the classes to be considered for training, each stored in a separate line!')
    a('--datasets', type=str, default='', help='Filename containing the names of datasets to be considered!')
    a('--model_save_name', type=str, default='', help='Filename for the saved model!')
    a('--settings_path', type=str, default='', help='Path to the training settings, stored in a .yml file')
    # Note : Output directory to follow the given structure :: Preprocessed -> <ClassName> -> <Parent Dataset> -> "all files here!"

    return parser.parse_args()

if __name__ == '__main__':
    arguments = get_arguments()
    classes = Reader.read_names(arguments.classes, 'class')
    datasets = Reader.read_names(arguments.datasets, 'dataset')
    input_dir = arguments.input_dir
    model_save_name = arguments.model_save_name
    assert input_dir != '', 'Input directory not provided!!'
    
    X, y = Data.get_input_files(input_dir, classes, datasets)
    X = Data.load_input_files(X, normalize=True)
    # input_data = [path.join(input_dir,x) for x in input_data]
    print ('Input Data >>>>>>')
    X = [x[:,:430] for x in X]
    print (len(X))
    print (X[0].shape)
    for x in X:
        print (x.shape)
    print (len(y))
    # print (y)
    X_temp = np.array(X)
    X_temp = np.expand_dims(X_temp,-1)

    enc = LabelBinarizer()
    enc.fit(y)
    y_temp = enc.transform(y)
    print (X_temp.shape)
    print (y_temp.shape)
    print (y_temp)
    train_model(X_temp,y_temp)
    # print (input_data)
    # parallelize(input_data, input_dir, settings, output_dir, n_jobs)
