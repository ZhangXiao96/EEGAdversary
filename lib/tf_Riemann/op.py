import tensorflow as tf


def logm(A):
    """
    This is especially used for Symmetric Matrix, the original "tf.linalg.logm"
    has no defined gradient operation.
    :param A:
    :return: Matrix Logarithm (A)
    """
    e, v = tf.linalg.eigh(A)
    e = tf.linalg.diag(tf.log(e))
    return tf.matmul(v, tf.matmul(e, tf.linalg.inv(v)))
