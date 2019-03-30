"""
This file is used to process data to a format which is better for the next experiments.
The original data format can be found in 'Data/original_data/data_description.pdf'.
After running this part, you will got two files in 'Data/processed_data/' for each subject.

*_test.mat and *_train.mat

Every .mat has 5 arrays.

signal: EEG signals, shape=(n_trial, n_channel, n_sample).
flashing: 1 when the intensification starts and otherwise 0, shape=(n_trial, n_sample)
stimuli: Which row/column intensifies, only valid on the sample where flashing==1,
         shape=(n_trial, n_sample). 1~12, actually flashing=np.where(stimuli!=0, 1, 0).
label: 1 for target and 0 for nonetarget, only valid on the sample where flashing==1,
         shape=(n_trial, n_sample)
char: True char for each trial, shape=(n_trial, )
"""

import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import butter, lfilter

# =============== parameters you may change ==============
subject = 'A'
save_dir = 'processed_data'

Fs = 240  # Hz
filter_high_cutoff = None  # Hz
filter_low_cutoff = None  # Hz
# ========================================================

TRAIN = 'Data/original_data/Subject_{}_Train.mat'.format(subject)
TEST = 'Data/original_data/Subject_{}_Test.mat'.format(subject)

if subject == 'A':
    TRUE_LABELS = list('WQXPLZCOMRKO97YFZDEZ1DPI9NNVGRQDJCUVRMEUOOOJD2UFYPOO6J7LDGYEGOA5VHNEHBTXOO1TDOILUEE5BFAEEXAW_K4R3MRU'.lower())
elif subject == 'B':
    TRUE_LABELS = list('MERMIROOMUHJPXJOHUVLEORZP3GLOO7AUFDKEFTWEOOALZOP9ROCGZET1Y19EWX65QUYU7NAK_4YCJDVDNGQXODBEV2B5EFDIDNR'.lower())

_CHAR_MATRIX = np.array(
            [list('abcdef'),
             list('ghijkl'),
             list('mnopqr'),
             list('stuvwx'),
             list('yz1234'),
             list('56789_')]
        )

# ========================= save train.mat ====================================
data = loadmat(TRAIN)

signal = data['Signal'].astype(np.float64)
flashing = data['Flashing'].astype(np.float64)
stimulusCode = data['StimulusCode'].astype(np.float64)
stimulusType = data['StimulusType'].astype(np.float64)
chars = list(data['TargetChar'][0])

n_trials = len(signal)
n_flashes = flashing.shape[1]
new_flashing = []
for trial in range(n_trials):
    flash_temp = []
    for flash in range(n_flashes):
        if flash == 0:
            flash_temp.append(1)
            continue
        if flashing[trial, flash] == 1 and flashing[trial, flash-1] == 0:
            flash_temp.append(1)
        else:
            flash_temp.append(0)
    flash_temp = np.reshape(np.array(flash_temp), newshape=[1, -1])
    new_flashing.append(flash_temp)

flashing = np.concatenate(new_flashing, axis=0)
stimuli = np.multiply(flashing, stimulusCode)
label = np.multiply(flashing, stimulusType)
chars = np.array(chars)
signal = np.transpose(signal, axes=[0, 2, 1])

# filter signal
fs_n = Fs / 2.
if filter_high_cutoff is not None:
    b, a = butter(5, [filter_low_cutoff / fs_n], btype='low')
    signal = lfilter(b, a, signal, axis=2)
if filter_low_cutoff is not None:
    b, a = butter(5, [filter_high_cutoff / fs_n], btype='High')
    signal = lfilter(b, a, signal, axis=2)

savemat(
    'Data/{}/{}_train.mat'.format(save_dir, subject),
    {'flashing': flashing, 'signal': signal, 'stimuli': stimuli, 'label': label, 'char': chars}
)

# ========================= save test.mat ====================================
data = loadmat(TEST)

signal = data['Signal'].astype(np.float64)
flashing = data['Flashing'].astype(np.float64)
stimulusCode = data['StimulusCode'].astype(np.float64)
chars = TRUE_LABELS

n_trials = len(signal)
n_flashes = flashing.shape[1]
new_flashing = []
for trial in range(n_trials):
    flash_temp = []
    for flash in range(n_flashes):
        if flash == 0:
            flash_temp.append(1)
            continue
        if flashing[trial, flash]==1 and flashing[trial, flash-1]==0:
            flash_temp.append(1)
        else:
            flash_temp.append(0)
    flash_temp = np.reshape(np.array(flash_temp), newshape=[1, -1])
    new_flashing.append(flash_temp)
flashing = np.concatenate(new_flashing, axis=0)
stimuli = np.multiply(flashing, stimulusCode)

label = []
for trial in range(n_trials):
    char = chars[trial]
    x, y = np.argwhere(_CHAR_MATRIX==char)[0]
    x += 7
    y += 1
    label.append(np.where(stimuli[trial:trial+1, :]==x, 1, 0) + np.where(stimuli[trial:trial+1, :]==y, 1, 0))

label = np.concatenate(label, axis=0)
chars = np.array(chars)
signal = np.transpose(signal, axes=[0, 2, 1])

# filter signal
fs_n = Fs / 2.
if filter_high_cutoff is not None:
    b, a = butter(5, [filter_low_cutoff / fs_n], btype='low')
    signal = lfilter(b, a, signal, axis=2)
if filter_low_cutoff is not None:
    b, a = butter(5, [filter_high_cutoff / fs_n], btype='High')
    signal = lfilter(b, a, signal, axis=2)

savemat(
    'Data/{}/{}_test.mat'.format(save_dir,subject),
    {'flashing': flashing, 'signal': signal, 'stimuli': stimuli, 'label': label, 'char': chars}
)