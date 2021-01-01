import os.path as path
import os
import argparse

import librosa
import numpy as np
import joblib
import read_config as Reader

def compute(x, input_dir, settings, output_dir):
    sr = settings['sample_rate']
    audio, sr = librosa.load(path.join(input_dir,x), sr)
    mels = librosa.feature.melspectrogram(
        audio,
        sr=sr,
        n_mels = settings['n_mels'],
        n_fft = settings['n_fft'],
        hop_length = settings['hop_length'],
        fmin = settings['fmin'],
        fmax = settings['fmax']
    )
    extension = x.split('.')[-1]
    save_file = x.replace(extension,'npz')
    np.savez(path.join(output_dir,save_file), mels)


def parallelize(input_data, input_dir, settings, output_dir, n_jobs):
    
    def gen_spec(x):
        return x, input_dir, settings, output_dir

    jobs = [joblib.delayed(compute)(*gen_spec(x)) for x in input_data]
    feature_files = joblib.Parallel(n_jobs=n_jobs, verbose=2)(jobs)

def get_arguments():
    parser = argparse.ArgumentParser()
    a = parser.add_argument

    a('--input_dir', type=str, default='', help='Input Directory containing audio files to be preprocessed!')
    a('--jobs', type=int, default=8, help='Number of parallel jobs!')
    a('--output_dir', type=str, default='', help='Output directory to store preprocessed inputs!')
    a('--settings_path', type=str, default='', help='Path to the preprocessing settings, stored in a .yml file')
    # Note : Output directory to follow the given structure :: Preprocessed -> <ClassName> -> <Parent Dataset> -> "all files here!"

    return parser.parse_args()

if __name__ == '__main__':
    arguments = get_arguments()
    settings = Reader.read_settings(arguments.settings_path, 'preprocess')
    input_dir = arguments.input_dir
    assert input_dir != '', 'Input directory not provided!!'
    output_dir = arguments.output_dir
    if not path.exists(output_dir):
        os.makedirs(output_dir)
    n_jobs = arguments.jobs

    input_data = os.listdir(input_dir)
    # input_data = [path.join(input_dir,x) for x in input_data]
    print ('Input Data >>>>>>')
    print (input_data)
    parallelize(input_data, input_dir, settings, output_dir, n_jobs)
