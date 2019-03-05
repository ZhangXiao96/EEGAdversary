from abc import abstractmethod

from mne.decoding import CSP as mne_CSP
from sklearn.decomposition import PCA as sklearn_PCA
from sklearn.decomposition import FastICA as sklearn_ICA
from pyriemann.tangentspace import TangentSpace as riemann_TangentSpace

from sklearn.linear_model import LogisticRegression as sklearn_LR
from pyriemann.classification import MDM as riemann_MDM

import tensorflow as tf
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Flatten as keras_Flatten

import numpy as np
from scipy.linalg import eig
from lib.tf_Riemann.Distance import distance
from lib.tf_Riemann.TangentSpace import tangent_space


class ProcessingBlock(object):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def get_keras_layer(self):
        pass


class ClassifierBlock(object):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def get_keras_layer(self):
        pass


# ------------------------Processors-----------------------
class Flatten(ProcessingBlock):
    def __init__(self, name="Flatten"):
        super(Flatten, self).__init__(name)


    def fit(self, x, y):
        pass

    def transform(self, x):
        n_epochs = x.shape[0]
        return np.reshape(x, newshape=(n_epochs, -1))

    def __get_weights(self):
        return None

    def get_keras_layer(self):
        return keras_Flatten()


class CSP(ProcessingBlock):
    def __init__(self, n_components=4, reg=None, log=None,
                 transform_into='csp_space', norm_trace=False, name="CSP"):
        super(CSP, self).__init__(name)
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.transform_into = transform_into
        self.norm_trace = norm_trace
        self.model = mne_CSP(n_components, reg, log, 'epoch', transform_into, norm_trace)

    def fit(self, x, y):
        assert len(x.shape) == 3
        self.model.fit(x, y)

    def transform(self, x):
        assert len(x.shape) == 3
        return self.model.transform(x)

    def __get_weights(self):
        csp_matrix = np.array(self.model.filters_[:self.model.n_components]).astype(np.float64)    # (n_components, channels)
        return csp_matrix

    def get_keras_layer(self):
        csp_matrix = self.__get_weights()

        def csp_transform(_x):
            csp_tensor = tf.constant(csp_matrix.T)  # (channels, n_components)
            channels, samples = _x.shape[1], _x.shape[2]
            _x = tf.reshape(_x, shape=(-1, 1, channels, samples))
            conv_filters = tf.reshape(csp_tensor, (csp_tensor.shape[0], 1, 1, self.n_components))
            _x = tf.nn.conv2d(_x, conv_filters, strides=[1, 1, 1, 1], padding='VALID', data_format='NCHW')
            _x = tf.reshape(_x, (-1, self.n_components, samples))
            # compute features (mean band power)
            if self.transform_into == 'average_power':
                _x = tf.reduce_mean(tf.square(_x), axis=2)
                log = True if self.log is None else self.log
                if log:
                    _x = tf.log(_x)
                else:
                    _x = _x - self.model.mean_
                    _x = _x / self.model.std_
            return _x

        return Lambda(csp_transform)


class PCA4Channel(ProcessingBlock):
    def __init__(self, n_components=4, whiten=True, name="PCA4Channel"):
        super(PCA4Channel, self).__init__(name)
        self.n_components = n_components
        self.whiten = whiten
        self.model = sklearn_PCA(n_components=self.n_components, whiten=self.whiten)

    def fit(self, x, y=None):
        assert len(x.shape) == 3
        _x = np.hstack(x)   # (n_channels, n_samples * n_epochs)
        self.model.fit(_x.T, y)

    def transform(self, x):
        assert len(x.shape) == 3
        n_epochs, n_channels, n_samples = x.shape
        _x = np.copy(x)
        _x = np.transpose(_x, axes=(1, 0, 2)).reshape((n_channels, n_epochs * n_samples))
        _x = self.model.transform(_x.T)
        _x = _x.T
        _x = _x.reshape((self.n_components, n_epochs, n_samples))
        _x = np.transpose(_x, axes=(1, 0, 2))
        return _x

    def __get_weights(self):
        pca_matrix = np.array(self.model.components_).astype(np.float64)
        mean = np.array(self.model.mean_).astype(np.float64)
        variance = np.array(self.model.explained_variance_).astype(np.float64)
        return mean, pca_matrix, variance

    def get_keras_layer(self):
        mean, pca_matrix, variance = self.__get_weights() # shape(pca_matrix) = (n_components, n_samples)
        if self.whiten:
            sqrt_variance = np.sqrt(variance)
            pca_matrix = pca_matrix / np.reshape(sqrt_variance, newshape=(self.n_components, 1))

        def pca_transform(_x):
            pca_tensor = tf.constant(pca_matrix.T)  # (channels, n_components)
            mean_tensor = tf.constant(mean)     # (n_components,)
            channels, samples = _x.shape[1], _x.shape[2]
            _x = tf.reshape(_x, shape=(-1, channels, 1, samples))
            _x = tf.nn.bias_add(_x, -mean_tensor, data_format='NCHW')
            _x = tf.reshape(_x, shape=(-1, 1, channels, samples))
            conv_filters = tf.reshape(pca_tensor, (pca_tensor.shape[0], 1, 1, self.n_components))
            _x = tf.nn.conv2d(_x, conv_filters, strides=[1, 1, 1, 1], padding='VALID', data_format='NCHW')
            _x = tf.reshape(_x, (-1, self.n_components, samples))
            return _x

        return Lambda(pca_transform)


class PCA4Feature(ProcessingBlock):
    def __init__(self, n_components=4, whiten=True, name="PCA4Feature"):
        super(PCA4Feature, self).__init__(name)
        self.n_components = n_components
        self.whiten = whiten
        self.model = sklearn_PCA(n_components=self.n_components, whiten=self.whiten)

    def fit(self, x, y=None):
        assert len(x.shape) >= 2    # (n_epochs, n_features)
        self.model.fit(x, y)

    def transform(self, x):
        assert len(x.shape) == 2
        _x = np.copy(x)
        _x = self.model.transform(_x)
        return _x

    def __get_weights(self):
        pca_matrix = np.array(self.model.components_).astype(np.float64)
        mean = np.array(self.model.mean_).astype(np.float64)
        variance = np.array(self.model.explained_variance_).astype(np.float64)
        return mean, pca_matrix, variance

    def get_keras_layer(self):
        mean, pca_matrix, variance = self.__get_weights() # shape(pca_matrix) = (n_components, n_samples)
        if self.whiten:
            sqrt_variance = np.sqrt(variance)
            pca_matrix = pca_matrix / np.reshape(sqrt_variance, newshape=(self.n_components, 1))

        def pca_transform(_x):
            pca_tensor = tf.constant(pca_matrix.T)  # (n_features, n_components)
            mean_tensor = tf.constant(mean)     # (n_components,)
            _x = tf.nn.bias_add(_x, -mean_tensor)
            _x = tf.matmul(_x, pca_tensor)
            return _x

        return Lambda(pca_transform)


class ICA(ProcessingBlock):
    def __init__(self, n_components=4,  max_iter=200, tol=1e-6, name="ICA"):
        super(ICA, self).__init__(name)
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.model = sklearn_ICA(n_components=self.n_components, whiten=True, tol=self.tol)

    def fit(self, x, y=None):
        assert len(x.shape) == 3
        _x = np.hstack(x)  # (n_channels, n_samples * n_epochs)
        self.model.fit(_x.T)

    def transform(self, x):
        assert len(x.shape) == 3
        n_epochs, n_channels, n_samples = x.shape
        _x = np.copy(x)
        _x = np.transpose(_x, axes=(1, 0, 2)).reshape((n_channels, n_epochs * n_samples))
        _x = self.model.transform(_x.T)
        _x = _x.T
        _x = _x.reshape((self.n_components, n_epochs, n_samples))
        _x = np.transpose(_x, axes=(1, 0, 2))
        return _x

    def __get_weights(self):
        unmixing_matrix = np.array(self.model.components_).astype(np.float64)
        mean_matrix = np.array(self.model.mean_).astype(np.float64)
        return unmixing_matrix, mean_matrix

    def get_keras_layer(self):
        unmixing_matrix, mean_matrix = self.__get_weights()

        def ica_transform(_x):
            ica_tensor = tf.constant(unmixing_matrix.T)  # (channels, n_components)
            channels, samples = _x.shape[1], _x.shape[2]
            _x = tf.reshape(_x, shape=(-1, channels, 1, samples))
            mean_tensor = tf.constant(mean_matrix)     # (n_components,)
            _x = tf.nn.bias_add(_x, -mean_tensor, data_format='NCHW')
            _x = tf.reshape(_x, shape=(-1, 1, channels, samples))
            conv_filters = tf.reshape(ica_tensor, (ica_tensor.shape[0], 1, 1, self.n_components))
            _x = tf.nn.conv2d(_x, conv_filters, strides=[1, 1, 1, 1], padding='VALID', data_format='NCHW')
            _x = tf.reshape(_x, (-1, self.n_components, samples))
            return _x

        return Lambda(ica_transform)


class CovarianceFeature(ProcessingBlock):
    def __init__(self, with_mean_templates=False, name="CovarianceFeature"):
        super(CovarianceFeature, self).__init__(name)
        self.with_templates = with_mean_templates

    def fit(self, x, y=None):
        assert len(x.shape) == 3
        if self.with_templates:
            assert y is not None, "\'y\' is needed to calculate the mean templates!"
            y_unique = np.unique(y)
            template_list = []
            for label in y_unique:
                template_list.append(np.mean(x[y == label, :, :], axis=0, keepdims=False))
            self.templates = np.concatenate(template_list, axis=0).astype(np.float64)

    def transform(self, x):
        assert len(x.shape) == 3        # (n_epochs, n_channels, n_samples)
        n_epochs, n_samples = x.shape[0], x.shape[2]
        if self.with_templates:
            templates = self.__get_weights()
            templates = templates[np.newaxis, :, :]
            templates = np.repeat(templates, repeats=n_epochs, axis=0)
            _x = np.concatenate((x, templates), axis=1)
        else:
            _x = x
        x_mean = np.mean(_x, axis=2, keepdims=True)
        _x = _x - x_mean
        _x_T = np.transpose(_x, axes=(0, 2, 1))  # (n_epochs, n_samples, n_channels)
        c = np.matmul(_x, _x_T)/(n_samples-1.)
        return c

    def __get_weights(self):
        templates = None
        if self.with_templates:
            templates = self.templates
        return templates

    def get_keras_layer(self):

        def cov_transform(_x):
            n_samples = int(_x.shape[2])
            if self.with_templates:
                templates_tensor = tf.constant(self.__get_weights())
                templates_tensor = tf.expand_dims(templates_tensor, 0)
                templates_tensor = tf.tile(templates_tensor, (tf.shape(_x)[0], 1, 1))
                _x = tf.concat((_x, templates_tensor), axis=1)
            _x_mean = tf.reduce_mean(_x, axis=2, keepdims=True)
            _x = _x - _x_mean
            _x_T = tf.transpose(_x, perm=(0, 2, 1))     # (n_epochs, n_samples, n_channel)
            c = tf.matmul(_x, _x_T) / (n_samples - 1.)
            return c

        return Lambda(cov_transform)


class Xdawn(ProcessingBlock):
    """
    Please Note that xDAWN is now just implemented to processing P300 EEG data which usually
    consists of 2 Classes (target and none-target). This implementation is based on EEG EEG
    epochs (the original implementation is based on EEG trials), which means we only need to
    calculate the spatial filters.
    Our implementation also references https://github.com/alexandrebarachant/bci-challenge-ner-2015
    which won the first prize at the BCI Challenge @ NER 2015 : https://www.kaggle.com/c/inria-bci-challenge.
    """
    def __init__(self, n_components=4, with_xdawn_templates=False, transform_flag=True, name="Xdawn"):
        """

        :param n_components: The number of spatial filters.
        :param with_xdawn_templates: Set True if padding the templates on the original EEG epochs.
               Usually used to calculate Xdawn Covariance Matrix.
        :param transform: Sometimes only the templates are needed, in this case set
               'transform=False' to just pad the templates on the original EEG epochs. Usually
               set to 'False' when using Xdawn Covariance Matrix.
        :param name: The name of the block.
        """
        super(Xdawn, self).__init__(name)
        self.n_components = n_components
        self.transform_flag = transform_flag
        self.with_templates = with_xdawn_templates

    def fit(self, x, y):
        assert len(x.shape) == 3    # (n_epochs, n_channels, n_samples)
        unique_y = np.unique(y)
        assert len(unique_y)==2 and set(unique_y).issubset(set([0,1])), \
            "Only P300 data is vailable, y=1 for target, while 0 for none-target!"

        P1 = np.mean(x[y == 1, :, :], axis=0)
        P0 = np.mean(x[y == 0, :, :], axis=0)

        C1 = np.matrix(np.cov(P1))
        C0 = np.matrix(np.cov(P0))
        _x = np.hstack(x)   # (n_channels, n_samples * n_epochs)
        Cx = np.matrix(np.cov(_x))

        # Spatial filters
        D, self.V1 = eig(C1, Cx)
        D, self.V0 = eig(C0, Cx)

        # create the templates (usually only used if covariance features are supposed to be used.)
        self.P = np.concatenate(
            (np.dot(self.V1[:, 0:self.n_components].T, P1), np.dot(self.V0[:, 0:self.n_components].T, P0)),
            axis=0
        )

    def transform(self, x):
        assert len(x.shape) == 3    # (n_epochs, n_channels, n_samples)
        n_epochs = x.shape[0]
        Xdawn_matrix, templates = self.__get_weights()
        Xdawn_matrix = Xdawn_matrix[np.newaxis, :, :]
        templates = templates[np.newaxis, :, :]
        if self.transform_flag:
            _x = np.matmul(Xdawn_matrix, x)
        else:
            _x = x
        if self.with_templates:
            templates = np.repeat(templates, repeats=n_epochs, axis=0)
            _x = np.concatenate((_x, templates), axis=1)
        return _x

    def __get_weights(self):
        Xdawn_matrix = np.array(self.V1[:self.n_components]).astype(np.float64)    # (n_components, channels)
        templates = self.P.astype(np.float64)
        return Xdawn_matrix, templates

    def get_keras_layer(self):
        Xdawn_matrix, templates = self.__get_weights()

        def xdawn_transform(_x):
            Xdawn_tensor = tf.constant(Xdawn_matrix.T)  # (channels, n_components)
            templates_tensor = tf.constant(templates)
            n_channels, n_samples = _x.shape[1], _x.shape[2]

            if self.transform_flag:
                _x = tf.reshape(_x, shape=(-1, 1, n_channels, n_samples))
                conv_filters = tf.reshape(Xdawn_tensor, (Xdawn_tensor.shape[0], 1, 1, self.n_components))
                _x = tf.nn.conv2d(_x, conv_filters, strides=[1, 1, 1, 1], padding='VALID', data_format='NCHW')
                _x = tf.reshape(_x, (-1, self.n_components, n_samples))

            if self.with_templates:
                _templates_tensor = tf.expand_dims(templates_tensor, 0)
                _templates_tensor = tf.tile(_templates_tensor, (tf.shape(_x)[0], 1, 1))
                _x = tf.concat((_x, _templates_tensor), axis=1)

            return _x

        return Lambda(xdawn_transform)


class TangentSpaceFeature(ProcessingBlock):
    def __init__(self, mean_metric='riemann', name="TangentSpaceFeature"):
        super(TangentSpaceFeature, self).__init__(name)
        self.mean_metric = mean_metric
        self.model = riemann_TangentSpace(metric=self.mean_metric, tsupdate=False)

    def fit(self, x, y):
        assert len(x.shape) == 3, 'The shape of \'x\' should be (n_epochs, n_channels, n_channles).'
        assert x.shape[1] == x.shape[2], 'The shape of \'x\' should be (n_epochs, n_channels, n_channles).'
        assert len(y) == len(x)
        self.model.fit(x, y)

    def transform(self, x):
        assert len(x.shape) == 3, 'The shape of \'x\' should be (n_epochs, n_channels, n_channles).'
        assert x.shape[1] == x.shape[2], 'The shape of \'x\' should be (n_epochs, n_channels, n_channles).'
        return self.model.transform(x)

    def __get_weights(self):
        Cref = np.array(self.model.reference_).astype(np.float64)
        return Cref

    def get_keras_layer(self):
        Cref = self.__get_weights()

        def tangent_space_transform(_x):
            Cref_tensor = tf.constant(Cref)  # (channels, channels)
            return tangent_space(_x, Cref_tensor)

        return Lambda(tangent_space_transform)


# ------------------------Classifiers-----------------------
class LogisticRegression(ClassifierBlock):
    def __init__(self, penalty='l2', tol=1e-6, C=1.0, class_weight=None, max_iter=100, name="LogisticRegression"):
        super(LogisticRegression, self).__init__(name)
        self.penalty = penalty
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.tol = tol
        self.model = sklearn_LR(penalty=self.penalty, tol=self.tol, C=self.C, solver='sag',
                                class_weight=self.class_weight, multi_class='multinomial')

    def fit(self, x, y=None):
        assert len(x.shape) == 2
        assert len(y) == len(x)
        self.model.fit(x, y)

    def predict(self, x):
        assert len(x.shape) == 2
        y = self.model.predict_proba(x)
        return y

    def __get_weights(self):
        coef = self.model.coef_
        intercept_ = self.model.intercept_
        if len(intercept_) == 1:
            """
            This is important! When len(class)==2, the Sklearn gives a different type of coef_
            and intercept. Look at Sklearn for more information!
            """
            coef = np.concatenate((-coef, coef), axis=0)
            intercept_ = np.concatenate((-intercept_, intercept_), axis=0)
        w = np.array(coef).astype(np.float64)
        b = np.array(intercept_).astype(np.float64)
        return w, b

    def get_keras_layer(self):
        w, b = self.__get_weights()  # shape(w) = (n_class, n_features)

        def LR_predict(_x):
            w_tensor = tf.constant(w.T)  # (n_features, n_class)
            b_tensor = tf.constant(b)  # (n_class,)
            _logits = tf.nn.xw_plus_b(_x, w_tensor, b_tensor)
            _probs = tf.nn.softmax(_logits)
            return _probs

        return Lambda(LR_predict)


class MDM(ClassifierBlock):
    """
    See https://pyriemann.readthedocs.io/en/latest/ for more information.
    """
    def __init__(self, mean_metric='riemann', dist_metric='riemann', n_jobs=1, name="MDM"):
        super(MDM, self).__init__(name)
        self.mean_metric = mean_metric
        self.dist_metric = dist_metric
        self.n_jobs = n_jobs
        metric = {'mean': self.mean_metric, 'distance': self.dist_metric}
        self.model = riemann_MDM(metric=metric, n_jobs=self.n_jobs)

    def fit(self, x, y):
        assert len(x.shape) == 3, 'The shape of \'x\' should be (n_epochs, n_channels, n_channles).'
        assert x.shape[1] == x.shape[2], 'The shape of \'x\' should be (n_epochs, n_channels, n_channles).'
        assert len(y) == len(x)
        self.model.fit(x, y)

    def predict(self, x):
        assert len(x.shape) == 3, 'The shape of \'x\' should be (n_epochs, n_channels, n_channles).'
        assert x.shape[1] == x.shape[2], 'The shape of \'x\' should be (n_epochs, n_channels, n_channles).'
        y = self.model.predict_proba(x)
        return y

    def __get_weights(self):
        covmeans = np.array(self.model.covmeans_).astype(np.float64)
        dist_metric = self.dist_metric
        return covmeans, dist_metric

    def get_keras_layer(self):
        covmeans, dist_metric = self.__get_weights()  # shape(w) = (n_class, n_channels, n_channels)

        def mdm_predict(_x):
            covmean_tensor_list = [tf.constant(covmeans[c, :, :]) for c in range(covmeans.shape[0])]
            dist_list = []
            n_epochs = tf.shape(_x)[0]
            for covmean_tensor in covmean_tensor_list:
                covmean_tensor = tf.expand_dims(covmean_tensor, 0)
                covmean_tensor = tf.tile(covmean_tensor, (n_epochs, 1, 1))
                dist = distance(_x, covmean_tensor, metric=dist_metric)
                dist = tf.reshape(dist, shape=(-1, 1))
                dist_list.append(dist)
            dist_tensor = tf.concat(dist_list, axis=1)
            _probs = tf.nn.softmax(-dist_tensor)
            return _probs
            # return dist_tensor

        return Lambda(mdm_predict)


if __name__ == '__main__':
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    x =  10+10*np.random.rand(10, 22, 250)
    x = x.astype(np.float64)
    x = np.matmul(x, np.transpose(x, axes=(0, 2, 1)))
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0])

    # model = CSP(n_components=4, transform_into='average_power')
    # model = PCA4Feature(n_components=3, whiten=True)
    # model = ICA(n_components=4)
    # model = Xdawn(n_components=4, with_templates=True, transform_flag=False)
    # model = CovarianceFeature(with_mean_templates=True)
    # model = MDM(dist_metric='riemann')
    # # 'riemann','logeuclid','euclid','logdet'
    model = TangentSpaceFeature()

    model.fit(x, y)
    z = model.transform(x)

    input_layer = Input(shape=(22, 22), dtype=tf.float64)
    features = model.get_keras_layer()(input_layer)
    keras_model = Model(inputs=input_layer, outputs=features)
    z2 = keras_model.predict(x)
    print(np.max(np.abs(z2-z)))