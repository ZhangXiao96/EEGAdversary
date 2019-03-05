import tensorflow as tf
from lib.tf_Riemann import op


def distance_euclid(A, B):
    return tf.norm(A-B, axis=(1, 2), ord='fro', keepdims=False)


def distance_logeuclid(A, B):
    A = op.logm(A)
    B = op.logm(B)
    return distance_euclid(A, B)


def distance_riemann(A, B):
    sqrt_B_1 = tf.linalg.inv(tf.linalg.sqrtm(B))
    m = tf.matmul(sqrt_B_1, A)
    m = tf.matmul(m, sqrt_B_1)
    m = op.logm(m)
    return tf.norm(m, axis=(-2, -1), keepdims=False)


def distance_logdet(A, B):
    m1 = tf.linalg.logdet((A+B)/2.)
    m2 = tf.linalg.logdet(A) + tf.linalg.logdet(B)
    m = tf.sqrt(m1-0.5 * m2)
    return m


distance_methods = {'riemann': distance_riemann,
                    'logeuclid': distance_logeuclid,
                    'euclid': distance_euclid,
                    'logdet': distance_logdet}


def distance(A, B, metric='riemann'):

    if callable(metric):
        distance_function = metric
    else:
        distance_function = distance_methods[metric]

    return distance_function(A, B)
