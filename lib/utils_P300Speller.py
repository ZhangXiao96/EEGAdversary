import numpy as np


_CHAR_MATRIX = np.array(
            [list('abcdef'),
             list('ghijkl'),
             list('mnopqr'),
             list('stuvwx'),
             list('yz1234'),
             list('56789_')]
        )


def get_start_indics(flashing):
    n_samples = len(flashing)
    start_indics = []
    for sample in range(n_samples):
        if sample == 0:
            start_indics.append(0)
            continue
        if flashing[sample] == 1 and flashing[sample - 1] == 0:
            start_indics.append(sample)
    return np.array(start_indics).squeeze()


def get_char(location, char_matrix=_CHAR_MATRIX):
    column, row = np.min(location), np.max(location)
    column -= 1
    row -= 7
    return char_matrix[row, column]


def get_locations(stimuli_code, stimuli_prob):
    prob_sum = np.zeros((12,))
    for i in range(12):
        prob_sum[i] = np.sum(stimuli_prob[stimuli_code == i+1])
    column = np.argmax(prob_sum[0:6]) + 1
    row = np.argmax(prob_sum[6:]) + 7
    return [column, row]


def add_template_noise(original_signal, stimuli, target_locations, to_target_template, perturb_length):
    """
    Add template noise to the character trial.
    :param original_signal: EEG trials. (n_trial, n_channel, n_sample).
    :param stimuli: stimuli code. (n_trial, n_sample).
    :param target_locations: the stimuli code of the target row and column. (row_code, column_code).
    :param to_target_template: the noise template. (n_channel, n_sample).
    :param perturb_length: Integer. The number of perturbation points.
    :return: adversarial signal
    """
    signal = np.copy(original_signal)
    n_trial = signal.shape[0]

    for i_trial in range(n_trial):
        start_ids = np.argwhere(stimuli[i_trial, :] != 0).ravel()
        temp_stimuli = stimuli[i_trial, start_ids].ravel()
        for sti in range(len(start_ids)):
            start_id = start_ids[sti]
            if temp_stimuli[sti] in target_locations:
                signal[i_trial, :, start_id:(start_id+perturb_length)] += to_target_template[:, :perturb_length]

    return signal


def trials_to_epochs(signal, label, stimuli, epoch_length):
    """
    Segment trials into epochs.
    :param signal: EEG signal. (n_trial, n_channel, n_sample).
    :param label: labels for epochs. 1 for "Target" and 0 for "Nontarget".
    :param stimuli: stimuli code. (n_trial, n_sample).
    :param epoch_length: The number of sample points in an epoch.
    :return: epochs and their labels.
    """
    n_trial = signal.shape[0]

    if label is not None:
        epochs = []
        y = []
        for i_trial in range(n_trial):
            start_ids = np.argwhere(stimuli[i_trial, :] != 0).ravel()
            for start_id in start_ids:
                epochs.append(signal[i_trial, :, start_id:(start_id + epoch_length)])
                y.append(label[i_trial, start_id])
        epochs = np.array(epochs).squeeze()
        y = np.array(y).ravel()
        return epochs, y
    else:
        epochs = []
        for i_trial in range(n_trial):
            start_ids = np.argwhere(stimuli[i_trial, :] != 0).ravel()
            for start_id in start_ids:
                epochs.append(signal[i_trial, :, start_id:(start_id + epoch_length)])
        epochs = np.array(epochs).squeeze()
        return epochs
