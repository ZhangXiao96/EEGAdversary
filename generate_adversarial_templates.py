"""
This file is used to generate and save adversarial templates with train set.
!!!NOTE: The templates did not times "epsilon", so l2_norm(each_channel) = 1.
!!!NOTE: Only "noise_to_target" template was used in future steps.
"""

import numpy as np
import lib.Blocks as Blocks
from lib.Pipeline import Pipeline
from lib.utils_P300Speller import trials_to_epochs
import scipy.io as io
from scipy import signal as sp_signal
from lib.KerasAdversary import WhiteBoxAttacks
import tensorflow.keras.backend as K
import tensorflow.keras.losses as losses
import os

K.set_floatx('float64')

# =============== parameters you may change ==============
subject = 'B'
window_time_length = 600  # ms
Fs = 240  # Hz
standard_before = True  # normalized before feature extraction
model_name = 'xDAWN+Riemann+LR'
data_dir = 'processed_data'
filter_high_cutoff = 15.  # Hz
filter_low_cutoff = 0.1  # Hz
# ========================================================

# filter for adversarial noise (build filters to filter the noise to [0.1, 60]Hz)
b, a = sp_signal.butter(4, [filter_low_cutoff/(Fs/2.), filter_high_cutoff/(Fs/2.)], 'bandpass')

model_dir = os.path.join('runs', model_name, subject)
load_path = os.path.join(model_dir, 'model.pkl')
other_path = os.path.join(model_dir, 'other_parameters.mat')
train_file = os.path.join('Data', data_dir, '{}_train.mat'.format(subject))
templates_path = os.path.join(model_dir, 'tampletes.mat')

_CHAR_MATRIX = np.array(
    [list('abcdef'),
     list('ghijkl'),
     list('mnopqr'),
     list('stuvwx'),
     list('yz1234'),
     list('56789_')]
)

# =============================== load data ===============================
data = io.loadmat(train_file)
signal = data['signal']
stimuli = data['stimuli']
label = data['label']
char = data['char']

n_trial, n_channel, n_sample = signal.shape[:]

# =========================== pre-processing data ========================
params = io.loadmat(other_path)
mean = params['mean']
std = params['std']
if standard_before:
    signal = (signal-mean)/std

epoch_length = int(Fs * window_time_length / 1000.)

# ============================ build model ==============================
processers = [
    Blocks.Xdawn(n_filters=8, with_xdawn_templates=True, apply_filters=True),
    Blocks.CovarianceFeature(with_mean_templates=False),
    Blocks.TangentSpaceFeature(mean_metric='riemann', name='TangentSpace({})'.format('riemann'))
]
classifier = Blocks.LogisticRegression()

model = Pipeline(processers, classifier)
model.load(load_path)
keras_model = model.get_keras_model(input_shape=(n_channel, epoch_length))
keras_model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

# ========================= split data ===========================
epochs, y = trials_to_epochs(signal, label, stimuli, epoch_length)

# ========================= building adversarial templates ==========================
attack_agent = WhiteBoxAttacks(keras_model, K.get_session(), loss_fn=losses.sparse_categorical_crossentropy)
adv_x = attack_agent.fgm(epochs, y, target=False, norm_ord=2, epsilon=1.)
adv_noise = adv_x - epochs

# to_target noise
noise_to_target = np.mean(adv_noise[y == 0, :, :], axis=0, keepdims=False)  # (channels, samples)
noise_to_target = sp_signal.filtfilt(b, a, noise_to_target)
noise_to_target = noise_to_target / np.linalg.norm(noise_to_target, axis=-1, keepdims=True)

# to_nontarget noise
noise_to_nonetarget = np.mean(adv_noise[y == 1, :, :], axis=0, keepdims=False)  # (channels, samples)
noise_to_nonetarget = sp_signal.filtfilt(b, a, noise_to_nonetarget)
noise_to_nonetarget = noise_to_nonetarget / np.linalg.norm(noise_to_nonetarget, axis=-1, keepdims=True)

io.savemat(templates_path, {'to_target': noise_to_target, 'to_nonetarget': noise_to_nonetarget})
