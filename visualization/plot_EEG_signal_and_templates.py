import numpy as np
import matplotlib.pyplot as plt
from lib.utils_P300Speller import add_template_noise
import scipy.io as io
import math
import os

# =============== parameters you may change ==============
# you should also change the pipline defined below to match the model which will be loaded.
# NOTE: interval is a very important parameter and relevant to data itself.
subject = 'A'
plot_start = 0
plot_time_length = 1000  # ms
plot_channel_ids = ['F3', 'F4', 'Cz', 'P3', 'P4']
# based on plotting channels, see 'visualization/matlab/eloc64.txt' for more information about locations.
plot_channel_indics = np.array([32, 36, 11, 49, 53]) - 1

interval = 175  # ms NOTE: interval should be no larger than the time between two intensifications.
standard_before = True  # normalized before feature extraction
model_name = 'xDAWN+Riemann+LR'
data_dir = 'processed_data'
original_char = 'y'
target_char = 'n'
epsilon = 0.5  # to control the SNR of EEG with noise.
Fs = 240  # Hz
# ========================================================

# figure setting (step between channels in the figure)
if subject == 'A':
    step = 3.
elif subject == 'B':
    step = 4.
else:
    raise Exception('No subject named \'{}\''.format(subject))

perturb_time = 2 * interval

_CHAR_MATRIX = np.array(
    [list('abcdef'),
     list('ghijkl'),
     list('mnopqr'),
     list('stuvwx'),
     list('yz1234'),
     list('56789_')]
)

model_dir = os.path.join('../runs', model_name, subject)
load_path = os.path.join(model_dir, 'model.pkl')
other_path = os.path.join(model_dir, 'other_parameters.mat')
test_file = os.path.join('../Data', data_dir, '{}_test.mat'.format(subject))
templates_path = os.path.join(model_dir, 'tampletes.mat')

# get templates
adv_templates = io.loadmat(templates_path)
to_target = adv_templates['to_target']


# =============================== load data ===============================
test_data = io.loadmat(test_file)
original_signal = test_data['signal']
stimuli = test_data['stimuli']
label = test_data['label']
char = test_data['char']

n_trial, n_channel, n_sample = original_signal.shape[:]

# pre-processing data
# get std and mean
params = io.loadmat(other_path)
mean = params['mean']
std = params['std']
if standard_before:
    original_signal = (original_signal-mean)/std


plot_length = int(Fs * plot_time_length / 1000.)
perturb_length = math.floor(Fs * perturb_time / 1000.)

x, y = np.argwhere(_CHAR_MATRIX == target_char)[0]
x += 7
y += 1
target_locations = (y, x)

# ============================ add adv_noise ============================
adv_signal = add_template_noise(original_signal, stimuli, target_locations, to_target, perturb_length, epsilon)

# ========================================================================
plot_trial_id = np.squeeze(np.argwhere(char==original_char)[0])
adv_x = adv_signal[plot_trial_id, :, :plot_length]
clean_x = original_signal[plot_trial_id, :, :plot_length]

# plot signals
plt.figure(figsize=[8, 4])
plt.subplot(121)
mean = np.mean(clean_x, axis=1, keepdims=True)
clean_x = clean_x - mean
adv_x = adv_x - mean

length = clean_x.shape[-1]
t = np.arange(length) * 1. / Fs

for j in range(len(plot_channel_indics)):
    plot_channel_index = plot_channel_indics[j]
    if j == 0:
        plt.plot(t, adv_x[plot_channel_index, :]+j*step, 'r', label='Adversarial trial (\'{}\')'.format(target_char.upper()), linewidth=2)
        plt.plot(t, clean_x[plot_channel_index, :]+j*step, 'g', label='Original trial (\'{}\')'.format(char[plot_trial_id].upper()), linewidth=2)
    else:
        plt.plot(t, adv_x[plot_channel_index, :]+j*step, 'r', linewidth=2)
        plt.plot(t, clean_x[plot_channel_index, :]+j*step, 'g', linewidth=2)
plt.xlabel('time (s)', fontsize=14)
plt.ylim([-step, 5*step + 2])
temp_y = np.arange(0, len(plot_channel_ids), 1) * step
plt.yticks(temp_y, plot_channel_ids, fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel('channel', fontsize=14)
plt.legend(loc='upper right', fontsize=10)
plt.title('Signals', fontsize=16)

# plot template
plt.subplot(122)
adv_templates = io.loadmat(templates_path)
perturb_length = math.floor(Fs * perturb_time / 1000.)
to_target = to_target[:, :perturb_length]
to_target = epsilon * to_target / np.linalg.norm(to_target, axis=-1, ord=2, keepdims=True)
to_target = to_target - np.mean(to_target, axis=-1, keepdims=True)

step = 0.2
t = np.arange(perturb_length) * 1. / Fs

for j in range(len(plot_channel_indics)):
    plot_channel_index = plot_channel_indics[j]
    plt.plot(t, to_target[plot_channel_index, :]+j*step, 'r', linewidth=2)

plt.xlabel('time (s)', fontsize=14)
plt.ylim([-0.5 * step, 4.5*step])
temp_y = np.arange(0, len(plot_channel_ids), 1) * step
plt.yticks(temp_y, plot_channel_ids, fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel('channel', fontsize=14)
plt.title('Template', fontsize=16)

plt.tight_layout()
plt.show()
