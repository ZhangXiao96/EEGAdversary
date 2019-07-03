"""
This file is used to simulate the test phase of P300 spellers.
"""

import numpy as np
import lib.Blocks as Blocks
from lib.Pipeline import Pipeline
import scipy.io as io
from lib.utils_P300Speller import get_char
from tqdm import tqdm
from scipy import signal as sp_signal
import os

# =============== parameters you may change ==============
# you should also change the pipline defined below to match the model which will be loaded.
subject = 'A'
window_time_length = 600  # ms
Fs = 240  # Hz
standard_before = True  # normalized before feature extraction (using mean and std in train set)
model_name = 'xDAWN+Riemann+LR'
data_dir = 'processed_data'
epsilon = 0.5  # to control the energy of the noise.
nb_rounds = 15
filter_high_cutoff = 15  # Hz, or None
filter_low_cutoff = 0.1  # Hz, or None
# ========================================================

# filter for random noise (build filters to filter the noise)
b, a = sp_signal.butter(4, [filter_low_cutoff/(Fs/2.), filter_high_cutoff/(Fs/2.)], 'bandpass')

model_dir = os.path.join('runs', model_name, subject)
load_path = os.path.join(model_dir, 'model.pkl')
other_path = os.path.join(model_dir, 'other_parameters.mat')
test_file = os.path.join('Data', data_dir, '{}_test.mat'.format(subject))

_CHAR_MATRIX = np.array(
            [list('abcdef'),
             list('ghijkl'),
             list('mnopqr'),
             list('stuvwx'),
             list('yz1234'),
             list('56789_')]
        )

# =============================== load data ===============================
test_data = io.loadmat(test_file)
signal = test_data['signal']
stimuli = test_data['stimuli']
label = test_data['label']
char = test_data['char']

n_trial, n_channel, n_sample = signal.shape[:]

# ======================== pre-processing data ========================
params = io.loadmat(other_path)
mean = params['mean']
std = params['std']
if standard_before:
    signal = (signal-mean)/std

length = int(Fs * window_time_length / 1000.)

# ============================ load model ==============================
processers = [
    Blocks.Xdawn(n_filters=8, with_xdawn_templates=True, apply_filters=True),
    Blocks.CovarianceFeature(with_mean_templates=False),
    Blocks.TangentSpaceFeature(mean_metric='riemann', name='TangentSpace({})'.format('riemann'))
]

classifier = Blocks.LogisticRegression()

model = Pipeline(processers, classifier)

# fixme:
# Now only load the weights for keras_model, so the model.predict() is not available.
# However, you could use keras_model.predict() instead.
model.load(load_path)
keras_model = model.get_keras_model(input_shape=(n_channel, length))

# =========================== test ================================
clean_chars = []
noisy_chars = []

for i_trial in tqdm(range(n_trial)):
    # get char prediction for each trial.
    epochs = []
    y = []
    start_ids = np.argwhere(stimuli[i_trial, :] != 0).ravel()[:nb_rounds*12]
    temp_stimuli = stimuli[i_trial, start_ids].ravel()
    for start_id in start_ids:
        epochs.append(signal[i_trial, :, start_id:(start_id+length)])
        y.append(label[i_trial, start_id])
    epochs = np.array(epochs).squeeze()
    y = np.array(y).ravel()

    # ====================== add clean target probs for each row/column =======================
    y_pred = keras_model.predict(epochs)[:, 1].ravel()
    unique_stimuli = np.unique(temp_stimuli)
    s_length = len(unique_stimuli)
    stimuli_pred = np.zeros(shape=(s_length, ))
    # simple voting of target probs
    for s in unique_stimuli:
        stimuli_pred[int(s-1)] = np.mean(y_pred[temp_stimuli == s])

    # get the prediction char
    location = (np.argsort(stimuli_pred[:6])[-1]+1, np.argsort(stimuli_pred[6:])[-1]+7)
    clean_chars.append(get_char(location))

    # ====================== add noisy target probs for each row/column ========================
    # add random noise
    noise = np.random.standard_normal(epochs.shape)
    noise = sp_signal.filtfilt(b, a, noise, axis=-1)  # the original data is filtered
    noise = noise / np.linalg.norm(noise, ord=2, axis=-1, keepdims=True)  # standard the noise with respect to channels
    noisy_epochs = epochs + epsilon * noise

    y_pred = keras_model.predict(noisy_epochs)[:, 1].ravel()
    noisy_stimuli_pred = np.zeros(shape=(s_length, ))
    # simple voting of target probs
    for s in unique_stimuli:
        noisy_stimuli_pred[int(s-1)] = np.mean(y_pred[temp_stimuli == s])

    # get the prediction char
    location = (np.argsort(noisy_stimuli_pred[:6])[-1]+1, np.argsort(noisy_stimuli_pred[6:])[-1]+7)
    noisy_chars.append(get_char(location))

# show the results
print()
print('Subject: {} (rounds={})'.format(subject, nb_rounds))
print('True  chars: {}'.format(''.join(char)))

print('Clean chars: {}'.format(''.join(clean_chars)))
clean_mark = np.where(clean_chars == char, 1, 0)
print('Clean marks: {}'.format(''.join([str(x) for x in clean_mark])))

print('Noisy chars: {}'.format(''.join(noisy_chars)))
noisy_mark = np.where(noisy_chars == char, 1, 0)
print('Noisy marks: {}'.format(''.join([str(x) for x in noisy_mark])))

print('Clean Acc: {}'.format(np.mean(clean_mark)))
print('Noisy Acc: {}'.format(np.mean(noisy_mark)))
