from lib.Pipeline import Pipeline
import lib.Blocks as Blocks
from lib import utils
import os
import numpy as np
from scipy.io import loadmat
from lib.KerasAdversary import WhiteBoxAttacks
import tensorflow.keras.backend as K
import tensorflow.keras.losses as losses
from prettytable import PrettyTable


K.set_floatx('float64')
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

data_dir = 'Data'
train_dir = 'runs'

data_name = 'MI4C'
s_num = 9
epsilon = 0.02

acc_matrix = np.zeros((3, s_num+1))
bca_matrix = np.zeros((3, s_num+1))

for s_id in range(s_num):
    # Build pathes
    data_path = os.path.join(data_dir, data_name, 's{}.mat'.format(s_id))

    # Load dataset
    data = loadmat(data_path)
    x_train = data['x_train'].squeeze()
    y_train = np.squeeze(data['y_train'])
    x_test = data['x_test'].squeeze()
    y_test = np.squeeze(data['y_test'])

    if 'ERN' in data_name or 'P300' in data_name or 'EPFL' in data_name:
        y0_rate = np.mean(np.where(y_train == 0, 1, 0))
        y1_rate = np.mean(np.where(y_train == 1, 1, 0))
        class_weight = {0: y1_rate, 1: y0_rate}
    else:
        class_weight = None

    data_size = y_train.shape[0]
    shuffle_index = utils.shuffle_data(data_size)
    x_train = x_train[shuffle_index]
    y_train = y_train[shuffle_index]

    nb_classes = len(np.unique(y_train))
    n_channel = x_train.shape[1]
    n_sample = x_train.shape[2]

    # Build Model
    processers = [
        Blocks.CSP(n_components=8, transform_into='csp_space', log=None, name='CSP(csp_space, {}->{})'.format(n_channel, 8)),
        Blocks.CovarianceFeature(with_mean_templates=False),
        Blocks.TangentSpaceFeature(mean_metric='riemann', name='TangentSpace({})'.format('riemman'))
    ]
    classifier = Blocks.LogisticRegression(class_weight=class_weight)
    # classifier = Blocks.MDM(dist_metric='logeuclid', n_jobs = nb_classes, name='MDM({})'.format('riemman'))
    # # 'riemann','euclid','logdet', 'logeuclid

    model = Pipeline(processers, classifier)
    print('subject {}: fitting.......'.format(s_id))
    model.fit(x_train, y_train)

    # clean
    y_pred = np.argmax(model.predict(x_test), axis=1)
    bca = utils.bca(y_test, y_pred)
    acc = np.sum(y_pred==y_test).astype(np.float64)/len(y_pred)
    acc_matrix[0, int(s_id)] = acc
    bca_matrix[0, int(s_id)] = bca

    # adversarial
    keras_model = model.get_keras_model(input_shape=(n_channel, n_sample))
    keras_model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    attack_agent = WhiteBoxAttacks(keras_model, K.get_session(), loss_fn=losses.sparse_categorical_crossentropy)
    adv_x = attack_agent.fgsm(x_test, y_pred, epsilon=epsilon)
    y_adv = np.argmax(model.predict(adv_x), axis=1)
    bca = utils.bca(y_test, y_adv)
    acc = np.sum(y_adv == y_test).astype(np.float64) / len(y_pred)
    acc_matrix[2, int(s_id)] = acc
    bca_matrix[2, int(s_id)] = bca

    # noisy
    x_noisy = x_test + epsilon * np.sign(np.random.standard_normal(x_test.shape))
    y_pred = np.argmax(model.predict(x_noisy), axis=1)
    bca = utils.bca(y_test, y_pred)
    acc = np.sum(y_pred==y_test).astype(np.float64)/len(y_pred)
    acc_matrix[1, int(s_id)] = acc
    bca_matrix[1, int(s_id)] = bca

    K.clear_session()

for n in range(3):
    acc_matrix[n, -1] = np.sum(acc_matrix[n, :-1])/s_num
    bca_matrix[n, -1] = np.sum(bca_matrix[n, :-1])/s_num

acc_matrix = np.round(acc_matrix, decimals=3)
bca_matrix = np.round(bca_matrix, decimals=3)

table1 = PrettyTable(['subject'] + list(range(1, 1+s_num)) + ['mean'])
table1.add_row(['clean']+list(acc_matrix[0]))
table1.add_row(['noisy']+list(acc_matrix[1]))
table1.add_row(['adversarial']+list(acc_matrix[2]))

table2 = PrettyTable(['subject'] + list(range(1, 1+s_num)) + ['mean'])
table2.add_row(['clean']+list(bca_matrix[0]))
table2.add_row(['noisy']+list(bca_matrix[1]))
table2.add_row(['adversarial']+list(bca_matrix[2]))

print('epsilon = {}'.format(epsilon))
model.pipeline_information()
print('RCA')
print(table1)
print('BCA')
print(table2)
