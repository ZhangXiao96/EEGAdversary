"""
This file was used to get averaged epochs for target and nontarget (real label) and also for
clean epochs and their adversarial epochs. This was necessary if you need to plot topomaps for ERP.

if you want to get averaged epochs for prediction label, see "get_averaged_epochs_on_pred_labels.py"
"""

import numpy as np
import scipy.io as io
import math
import os

# =============== parameters you may change ==============
# you should also change the pipline defined below to match the model which will be loaded.
# NOTE: interval is a very important parameter and relevant to data itself.
subject = 'B'
window_time_length = 600  # ms
interval = 175  # ms NOTE: interval should be no larger than the time between two intensifications.
standard_before = True  # normalized before feature extraction
model_name = 'xDAWN+Riemann+LR'
data_dir = 'processed_data'
target_char = 'z'
epsilon = 0.6  # to control the SNR of EEG with noise indirectly.
Fs = 240  # Hz
# ========================================================

perturb_time = 2 * interval

_CHAR_MATRIX = np.array(
    [list('abcdef'),
     list('ghijkl'),
     list('mnopqr'),
     list('stuvwx'),
     list('yz1234'),
     list('56789_')]
)

model_dir = os.path.join('runs', model_name, subject)
load_path = os.path.join(model_dir, 'model.pkl')
other_path = os.path.join(model_dir, 'other_parameters.mat')
test_file = os.path.join('Data', data_dir, '{}_test.mat'.format(subject))
templates_path = os.path.join(model_dir, 'tampletes.mat')
avg_epoch_path = os.path.join(model_dir, 'avg_epochs_real.mat')

# get templates
adv_templates = io.loadmat(templates_path)
to_target = epsilon * adv_templates['to_target']

# =============================== load data ===============================
test_data = io.loadmat(test_file)
original_signal = test_data['signal']
stimuli = test_data['stimuli']
label = test_data['label']
char = test_data['char']

n_trial, n_channel, n_sample = original_signal.shape[:]

# pre-processing data
# get std and mean
if standard_before:
    params = io.loadmat(other_path)
    mean = params['mean']
    std = params['std']
    original_signal = (original_signal-mean)/std


epoch_length = int(Fs * window_time_length / 1000.)
perturb_length = math.floor(Fs * perturb_time / 1000.)

signal = np.copy(original_signal)
x, y = np.argwhere(_CHAR_MATRIX == target_char)[0]
x += 7
y += 1
target_locations = (y, x)

# ============================ add adv_noise and split to epochs============================

adv_epochs = []
clean_epochs = []
real_y = []

for i_trial in range(n_trial):

    adv_epochs_trial = []
    clean_epochs_trial = []
    real_y_trial = []

    # add adv_noise
    start_ids = np.argwhere(stimuli[i_trial, :] != 0).ravel()
    temp_stimuli = stimuli[i_trial, start_ids].ravel()
    for sti in range(len(start_ids)):
        start_id = start_ids[sti]
        if temp_stimuli[sti] in target_locations:
            signal[i_trial, :, start_id:(start_id + perturb_length)] \
                += to_target[:, :perturb_length]

    # split data into epochs
    for start_id in start_ids:
        adv_epochs_trial.append(signal[i_trial, :, start_id:(start_id+epoch_length)])
        clean_epochs_trial.append(original_signal[i_trial, :, start_id:(start_id+epoch_length)])
        real_y_trial.append(label[i_trial, start_id])

    adv_epochs_trial = np.array(adv_epochs_trial).squeeze()  # (n_epochs_a_trial, n_channels, n_samples)
    clean_epochs_trial = np.array(clean_epochs_trial).squeeze()  # (n_epochs_a_trial, n_channels, n_samples)
    real_y_trial = np.array(real_y_trial).ravel()  # (n_epochs_a_trial,)

    adv_epochs.append(adv_epochs_trial)
    clean_epochs.append(clean_epochs_trial)
    real_y.append(real_y_trial)

adv_epochs = np.array(adv_epochs).squeeze()  # (n_trials, n_epochs_a_trial, n_channels, n_samples)
clean_epochs = np.array(clean_epochs).squeeze()  # (n_trials, n_epochs_a_trial, n_channels, n_samples)
real_y = np.array(real_y)  # (n_trials, n_epochs_a_trial)
# ============================================================================================

# =========================== averaging target and none target for each trial ========================
# (n_trials, n_channel, n_samples)
clean_avg_epoch_target = \
    np.array([np.mean(clean_epochs[i, real_y[i, :]==1, :, :], axis=0, keepdims=False) for i in range(n_trial)])
clean_avg_epoch_nontarget = \
    np.array([np.mean(clean_epochs[i, real_y[i, :]==0, :, :], axis=0, keepdims=False) for i in range(n_trial)])

adv_avg_epoch_target = \
    np.array([np.mean(adv_epochs[i, real_y[i, :]==1, :, :], axis=0, keepdims=False) for i in range(n_trial)])
adv_avg_epoch_nontarget = \
    np.array([np.mean(adv_epochs[i, real_y[i, :]==0, :, :], axis=0, keepdims=False) for i in range(n_trial)])

io.savemat(
    avg_epoch_path,
    {
        'clean_target': clean_avg_epoch_target,
        'clean_nontarget': clean_avg_epoch_nontarget,
        'adv_target': adv_avg_epoch_target,
        'adv_nontarget': adv_avg_epoch_nontarget,
        'target_char': target_char
    }
)

