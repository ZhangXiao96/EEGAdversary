import numpy as np
import math
from sklearn.metrics import confusion_matrix


def bca(y_true, y_pred):
    m = confusion_matrix(y_true, y_pred)
    numb = m.shape[0]
    acc_each_label = 0
    for i in range(numb):
        acc = m[i, i]/np.sum(m[i, :], keepdims=False).astype(np.float32)
        acc_each_label += acc
    return acc_each_label/numb


def ITR(P, Q, T):
    """
    Calculate the Information Translate Rate (ITR).
    ITR=\frac{60}{T}[log_2Q+Plog_2P+(1-P)log_2\frac{1-P}{Q-1}]
    :param P: Accuracy.
    :param Q: Number of targets (classes).
    :param T: Used time to translate a target. Unit: s.
    :return: ITR. Unit: bits/min.
    """
    return 60./T * (math.log2(Q)+P*math.log2(P)+(1-P)*math.log2((1.-P)/(Q-1.)))


def acc(y_true, y_pred):
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()
    return np.sum(y_pred == y_true).astype(np.float64) / len(y_pred)


def batch_iter(data, batchsize, shuffle=True):
    data = np.array(list(data))
    data_size = data.shape[0]
    num_batches = int((data_size-1)/batchsize) + 1
    # Shuffle the data
    if shuffle:
        shuffle_indices = shuffle_data(data_size)
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches):
        start_index = batch_num * batchsize
        end_index = min((batch_num + 1) * batchsize, data_size)
        yield shuffled_data[start_index:end_index]


def get_split_indices(data_size, split=[9, 1], shuffle=True):
    if len(split) < 2:
        raise TypeError('The length of split should be larger than 2 while the length of your split is {}!'.format(len(split)))
    split = np.array(split)
    split = split / np.sum(split)
    if shuffle:
        indices = shuffle_data(data_size)
    else:
        indices = np.arange(data_size)
    split_indices_list = []
    start = 0
    for i in range(len(split)-1):
        end = start + int(np.floor(split[i] * data_size))
        split_indices_list.append(indices[start:end])
        start = end
    split_indices_list.append(indices[start:])
    return split_indices_list


def shuffle_data(data_size, random_seed=None):
    if random_seed:
        np.random.seed(random_seed)
    indices = np.arange(data_size)
    return np.random.permutation(indices).squeeze()
