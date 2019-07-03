"""
This file is used to train the model, and save the model and
some other parameters, e.g., std and mean, in 'runs/$model_name$/'.
"""
import numpy as np
import lib.Blocks as Blocks
from lib.Pipeline import Pipeline
from lib.utils_P300Speller import trials_to_epochs
import scipy.io as io
import lib.utils as utils
import os

# =============== parameters you may change ==============
# you can also change the pipeline defined below.
subject = 'A'
window_time_length = 600  # ms
Fs = 240  # Hz
apply_class_weight = True
standard_before = True  # normalized before feature extraction
model_name = 'xDAWN+Riemann+LR'
data_dir = 'processed_data'
# ========================================================

model_dir = os.path.join('runs', model_name, subject)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

save_path = os.path.join(model_dir, 'model.pkl')
other_path = os.path.join(model_dir, 'other_parameters.mat')
train_file = os.path.join('Data', data_dir, '{}_train.mat'.format(subject))

_CHAR_MATRIX = np.array(
            [list('abcdef'),
             list('ghijkl'),
             list('mnopqr'),
             list('stuvwx'),
             list('yz1234'),
             list('56789_')]
        )

# ============================== load data ==================================
train_data = io.loadmat(train_file)
signal = train_data['signal']
stimuli = train_data['stimuli']
label = train_data['label']
char = train_data['char']

n_trial, n_channel, n_sample = signal.shape[:]

# =============== save mean and std with respect to channels =================
temp = np.reshape(np.transpose(signal, axes=[1, 2, 0]), newshape=(n_channel, -1))  # concat to be (channels, sample * trial)
mean = np.mean(temp, axis=1, keepdims=True)
std = np.std(temp, axis=1, keepdims=True)
mean = mean[np.newaxis, :, :]  # (1, n_channel, 1), this is better for broadcasting in numpy
std = std[np.newaxis, :, :]  # (1, n_channel, 1), this is better for broadcasting in numpy
io.savemat(other_path, {'mean': mean, 'std': std})

# =============== preprocessing =====================
if standard_before:
    signal = (signal-mean)/std

# data segment
epoch_length = int(Fs * window_time_length / 1000.)
epochs, y = trials_to_epochs(signal, label, stimuli, epoch_length)

# balance data
if apply_class_weight:
    y0_rate = np.mean(np.where(y == 0, 1, 0))
    y1_rate = np.mean(np.where(y == 1, 1, 0))
    class_weight = {0: y1_rate, 1: y0_rate}
else:
    class_weight = None

# ============================== build model ==================================
processers = [
    Blocks.Xdawn(n_filters=8, with_xdawn_templates=True, apply_filters=True),
    Blocks.CovarianceFeature(with_mean_templates=False),
    Blocks.TangentSpaceFeature(mean_metric='riemann', name='TangentSpace({})'.format('riemann'))
]

classifier = Blocks.LogisticRegression(class_weight=class_weight, max_iter=2000)

model = Pipeline(processers, classifier)
print('subject: {}'.format(subject))
model.pipeline_information()  # show model information

# ============================== train model ===================================
print('fitting.......')
model.fit(epochs, y)
model.save(save_path)

y_pred = np.argmax(model.predict(epochs), axis=1)
bca = np.round(utils.bca(y, y_pred), decimals=3)
acc = np.round(np.sum(y_pred == y).astype(np.float64)/len(y_pred), decimals=3)
print('train (target and nontarget): acc={}, bca={}'.format(acc, bca))
