"""
This file is to test the target attack with adversarial templates.
NOTE:!!!!!!!!!!!!!
The interval of two intensification (two adjacent intensification are not the same) is 175ms
(100ms for intensification and 75 ms for rest), but the length of an epoch is 600ms.
Here we only add noise to the epochs which we wanted to be target, which means we could only
add noise on the epoch[:2*175ms] (times 2 because two adjacent intensification are not the same),
and the noise added on epoch[2*175ms:600ms] should depend on the following stimuli. You can find this in
'add adv_noise' part below.

For example:
time:                  0 ms            175ms           350 ms           525 ms
stimuli:                1                5               7                3
signal:                 |----------------|---------------|----------------|
target label you want:  0                1               0                0
add adv templates      None      to_1 (last 2*175ms)     -               None

where 'to_1' means 'to_target_templates[:2*interval]'

after doing this on all the trial, the segment is performed.
"""

import numpy as np
import lib.Blocks as Blocks
from lib.Pipeline import Pipeline
from lib.utils_P300Speller import add_template_noise
import scipy.io as io
from lib.utils_P300Speller import get_char
import math
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# =============== parameters you may change ==============
# you should also change the pipline defined below to match the model which will be loaded.
# NOTE: interval is a very important parameter and relevant to data itself.
subject = 'B'
window_time_length = 600  # ms
interval = 175  # ms. The time between two intensifications.
standard_before = True  # normalized before feature extraction
model_name = 'xDAWN+Riemann+LR'
data_dir = 'processed_data'
acc_file_name = 'attack_acc'
target_char_list = list('abcdefghijklmnopqrstuvwxyz123456789_')  # to perturb all 100 trials to the target char.
epsilon = 0.6  # to control the SNR of EEG with noise indirectly.
Fs = 240  # Hz
# ========================================================
perturb_time = 2 * interval

model_dir = os.path.join('runs', model_name, subject)
load_path = os.path.join(model_dir, 'model.pkl')
other_path = os.path.join(model_dir, 'other_parameters.mat')
test_file = os.path.join('Data', data_dir, '{}_test.mat'.format(subject))
templates_path = os.path.join(model_dir, 'tampletes.mat')
acc_file_path = os.path.join(model_dir, '{}.npz'.format(acc_file_name))


_CHAR_MATRIX = np.array(
            [list('abcdef'),
             list('ghijkl'),
             list('mnopqr'),
             list('stuvwx'),
             list('yz1234'),
             list('56789_')]
        )

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

# =========================== target attack ==============================
acc_dict = {}
for target_char in target_char_list:

    # get stimuli of the target char
    x, y = np.argwhere(_CHAR_MATRIX == target_char)[0]
    x += 7
    y += 1
    target_locations = (y, x)

    # ============================ add adv_noise ============================
    adv_signal = add_template_noise(original_signal, stimuli, target_locations, to_target, perturb_length)

    # =========================== test ================================
    predict_chars = []
    for i_trial in range(n_trial):
        epochs = []
        y = []
        start_ids = np.argwhere(stimuli[i_trial, :] != 0).ravel()
        temp_stimuli = stimuli[i_trial, start_ids].ravel()

        # split the char trial to epochs
        for start_id in start_ids:
            epochs.append(adv_signal[i_trial, :, start_id:(start_id+epoch_length)])
            y.append(label[i_trial, start_id])
        epochs = np.array(epochs).squeeze()  # (n_epochs, n_channels, n_samples)
        y = np.array(y).ravel()  # (n_epochs,)

        # get target probs
        y_pred = keras_model.predict(epochs)[:, 1].ravel()

        # simple voting
        unique_stimuli = np.unique(temp_stimuli)
        s_length = len(unique_stimuli)
        stimuli_pred = np.zeros(shape=(s_length, ))
        for s in unique_stimuli:
            stimuli_pred[int(s-1)] = np.mean(y_pred[temp_stimuli == s])

        location = (np.argsort(stimuli_pred[:6])[-1]+1, np.argsort(stimuli_pred[6:])[-1]+7)
        predict_chars.append(get_char(location))

    # print results for each target character
    print('Target: {}'.format(target_char))
    print('True chars: {}'.format(''.join(char)))
    print('Pred chars: {}'.format(''.join(predict_chars)))
    true_mark = np.where(np.array(predict_chars)==target_char, 1, 0)
    print('Tran marks: {}'.format(''.join([str(x) for x in true_mark])))
    acc = np.mean(true_mark)
    print('Attacking success rate: {}'.format(acc))
    acc_dict[target_char] = acc

np.savez(acc_file_path, acc=acc_dict)
