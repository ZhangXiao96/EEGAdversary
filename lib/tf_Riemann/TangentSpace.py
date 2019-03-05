import tensorflow as tf
import numpy as np
from lib.tf_Riemann import op
from tensorflow.python.ops.distributions.util import fill_triangular_inverse


def tangent_space(covmats, Cref):
    """Project a set of covariance matrices in the tangent space according to the given reference point Cref

    :param covmats: Covariance matrices set, Ntrials X Nchannels X Nchannels
    :param Cref: The reference covariance matrix
    :returns: the Tangent space , a matrix of Ntrials X (Nchannels*(Nchannels+1)/2)

    """
    n_epochs = tf.shape(covmats)[0]
    n_channel = covmats.shape[1]
    Cm12 = tf.linalg.inv(tf.linalg.sqrtm(Cref))
    Cm12 = tf.expand_dims(Cm12, 0)
    Cm12 = tf.tile(Cm12, (n_epochs, 1, 1))
    m = tf.matmul(Cm12, covmats)
    m = tf.matmul(m, Cm12)
    m = op.logm(m)
    coeffs = (np.sqrt(2) * np.triu(np.ones((n_channel, n_channel)), 1) +np.eye(n_channel))
    coeffs_tensor = tf.constant(coeffs)
    coeffs_tensor = tf.expand_dims(coeffs_tensor, 0)

    m = tf.multiply(m, coeffs_tensor)
    T_list = []
    for i in range(n_channel):
        T_list.append(tf.reshape(m[:, i, i:], shape=(n_epochs, -1)))
    T = tf.concat(T_list, axis=1)
    return T