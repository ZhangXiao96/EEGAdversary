"""
This file is mainly to re-implement the traditional blocks in Tensorflow,
so that the gradients could be calculate easily.

processing blocks: Normalizer, Flatten, CSP, PCA(features or channels), ICA, xDAWN
                   Covariance, TangentSpace('riemann' , 'logeuclid' , 'euclid' , 'logdet')

classifier blocks: Logistic Regression, LinearSVM, MDM('riemann' , 'logeuclid' , 'euclid' , 'logdet'), LDA
"""

from abc import abstractmethod

from mne.decoding import CSP as mne_CSP
from sklearn.decomposition import PCA as sklearn_PCA
from sklearn.decomposition import FastICA as sklearn_ICA
from sklearn.preprocessing import Normalizer as sklearn_Normalizer
from pyriemann.tangentspace import TangentSpace as riemann_TangentSpace
from pyriemann.spatialfilters import Xdawn as riemman_Xdawn

from sklearn.linear_model import LogisticRegression as sklearn_LR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as sklearn_LDA
from sklearn.svm import LinearSVC as sklearn_LSVC
from pyriemann.classification import MDM as riemann_MDM

import tensorflow as tf
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Flatten as keras_Flatten

import numpy as np
from lib.tf_Riemann.Distance import distance
from lib.tf_Riemann.TangentSpace import tangent_space


class ProcessingBlock(object):
    def __init__(self, name):
        self.name = name
        self.weights = []

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def get_keras_layer(self):
        pass

    def get_weights(self):
        return self.weights

    def load_weights(self, weights):
        self.weights = weights


class ClassifierBlock(object):
    def __init__(self, name):
        self.name = name
        self.weights = []

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def get_keras_layer(self):
        pass

    def get_weights(self):
        return self.weights

    def load_weights(self, weights):
        self.weights = weights


# -----------------------ProcessingBlock------------------
class Normalizer(ProcessingBlock):
    def __init__(self, norm='l1', name="Normalizer"):
        super(Normalizer, self).__init__(name)
        self.norm = norm
        self.model = sklearn_Normalizer(norm=self.norm)

    def fit(self, x, y):
        pass

    def transform(self, x):
        return self.model.transform(x)

    def get_keras_layer(self):

        def norm_transform(_x):
            if self.norm == 'l1':
                norms = tf.norm(_x, ord=1, axis=1, keepdims=True)
            elif self.norm == 'l2':
                norms = tf.norm(_x, ord=2, axis=1, keepdims=True)
            else:
                raise Exception('\'{}\' is not available, should be in (\'l1\', \'l2\').'.format(self.norm))
            return _x / norms

        return Lambda(norm_transform)


class Flatten(ProcessingBlock):
    def __init__(self, name="Flatten"):
        super(Flatten, self).__init__(name)

    def fit(self, x, y):
        pass

    def transform(self, x):
        n_epochs = x.shape[0]
        return np.reshape(x, newshape=(n_epochs, -1))

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
        self.weights = [np.array(self.model.filters_[:self.model.n_components]).astype(np.float64)]

    def transform(self, x):
        assert len(x.shape) == 3
        return self.model.transform(x)

    def get_keras_layer(self):
        csp_matrix = self.weights[0]

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
        pca_matrix = np.array(self.model.components_).astype(np.float64)
        mean = np.array(self.model.mean_).astype(np.float64)
        variance = np.array(self.model.explained_variance_).astype(np.float64)
        self.weights = [mean, pca_matrix, variance]

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

    def get_keras_layer(self):
        mean, pca_matrix, variance = self.weights # shape(pca_matrix) = (n_components, n_samples)
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
        pca_matrix = np.array(self.model.components_).astype(np.float64)
        mean = np.array(self.model.mean_).astype(np.float64)
        variance = np.array(self.model.explained_variance_).astype(np.float64)
        self.weights = [mean, pca_matrix, variance]

    def transform(self, x):
        assert len(x.shape) == 2
        _x = np.copy(x)
        _x = self.model.transform(_x)
        return _x

    def get_keras_layer(self):
        mean, pca_matrix, variance = self.weights # shape(pca_matrix) = (n_components, n_samples)
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
        unmixing_matrix = np.array(self.model.components_).astype(np.float64)
        mean_matrix = np.array(self.model.mean_).astype(np.float64)
        self.weights = [unmixing_matrix, mean_matrix]

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

    def get_keras_layer(self):
        unmixing_matrix, mean_matrix = self.weights

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
        """
        Covariance Matrix of the inputs.
        :param with_mean_templates: boolean. Always set to True when decoding ERP data. However, you should be
        ware that Xdawn block have already added mean templates.
        :param name: The name of the block.
        """
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
            templates = np.concatenate(template_list, axis=0).astype(np.float64)
            self.weights = [templates]

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

    def get_keras_layer(self):

        def cov_transform(_x):
            n_samples = int(_x.shape[2])
            if self.with_templates:
                t = self.weights[0]
                templates_tensor = tf.constant(t)
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
    def __init__(self, n_filters=4, with_xdawn_templates=False, apply_filters=True, name="Xdawn"):
        """

        :param n_filters: The number of spatial filters. When "transform_flag=False", the original_data would
               not be filtered.
        :param with_xdawn_templates: Set True if padding the templates on the original EEG epochs.
               Usually used to calculate Xdawn Covariance Matrix.
        :param apply_filters: Sometimes only the templates are needed, in this case set
               'transform=False' to just pad the templates on the original EEG epochs. Usually
               set to 'False' when using Xdawn Covariance Matrix.
        :param name: The name of the block.
        """
        super(Xdawn, self).__init__(name)
        self.n_filters = n_filters
        self.apply_filters = apply_filters
        self.with_templates = with_xdawn_templates
        self.model = riemman_Xdawn(nfilter=self.n_filters)

    def fit(self, x, y):
        assert len(x.shape) == 3    # (n_epochs, n_channels, n_samples)
        self.model.fit(x, y)
        Xdawn_matrix = np.array(self.model.filters_).astype(np.float64)    # (n_filters * n_classes, channels)
        templates = np.array(self.model.evokeds_)
        n_components = len(self.model.classes_) * self.n_filters
        self.weights = [Xdawn_matrix, templates, n_components]

    def transform(self, x):
        assert len(x.shape) == 3    # (n_epochs, n_channels, n_samples)
        _x = x
        n_epochs = x.shape[0]
        if self.apply_filters:
            _x = self.model.transform(_x)
        if self.with_templates:
            _, templates_origin, _ = self.weights
            templates = np.copy(templates_origin)
            templates = templates[np.newaxis, :, :]
            templates = np.repeat(templates, repeats=n_epochs, axis=0)
            _x = np.concatenate((_x, templates), axis=1)
        return _x

    def get_keras_layer(self):
        Xdawn_matrix, templates, n_components = self.weights

        def xdawn_transform(_x):
            Xdawn_tensor = tf.constant(Xdawn_matrix.T)  # (channels, n_components)
            templates_tensor = tf.constant(templates)
            n_channels, n_samples = _x.shape[1], _x.shape[2]

            if self.apply_filters:
                _x = tf.reshape(_x, shape=(-1, 1, n_channels, n_samples))
                conv_filters = tf.reshape(Xdawn_tensor, (Xdawn_tensor.shape[0], 1, 1, n_components))
                _x = tf.nn.conv2d(_x, conv_filters, strides=[1, 1, 1, 1], padding='VALID', data_format='NCHW')
                _x = tf.reshape(_x, (-1, n_components, n_samples))

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
        Cref = np.array(self.model.reference_).astype(np.float64)
        self.weights = [Cref]

    def transform(self, x):
        assert len(x.shape) == 3, 'The shape of \'x\' should be (n_epochs, n_channels, n_channles).'
        assert x.shape[1] == x.shape[2], 'The shape of \'x\' should be (n_epochs, n_channels, n_channles).'
        return self.model.transform(x)

    def get_keras_layer(self):
        Cref = self.weights[0]

        def tangent_space_transform(_x):
            Cref_tensor = tf.constant(Cref)  # (channels, channels)
            return tangent_space(_x, Cref_tensor)

        return Lambda(tangent_space_transform)


# ------------------------Classifiers-----------------------
class LogisticRegression(ClassifierBlock):
    def __init__(self, penalty='l2', tol=1e-6, C=1.0, class_weight=None, max_iter=500, name="LogisticRegression"):
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
        coef = self.model.coef_
        intercept_ = self.model.intercept_
        if len(intercept_) == 1:
            """
            This is important! When len(class)==2, Sklearn gives a different type of coef_
            and intercept. Look at Sklearn for more information!
            """
            coef = np.concatenate((-coef, coef), axis=0)
            intercept_ = np.concatenate((-intercept_, intercept_), axis=0)
        w = np.array(coef).astype(np.float64)
        b = np.array(intercept_).astype(np.float64)
        self.weights = [w, b]

    def predict(self, x):
        assert len(x.shape) == 2
        y = self.model.predict_proba(x)
        return y

    def get_keras_layer(self):
        w, b = self.weights  # shape(w) = (n_class, n_features)

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
        covmeans = np.array(self.model.covmeans_).astype(np.float64)
        dist_metric = self.dist_metric
        self.weights = [covmeans, dist_metric]

    def predict(self, x):
        assert len(x.shape) == 3, 'The shape of \'x\' should be (n_epochs, n_channels, n_channles).'
        assert x.shape[1] == x.shape[2], 'The shape of \'x\' should be (n_epochs, n_channels, n_channles).'
        y = self.model.predict_proba(x)
        return y

    def get_keras_layer(self):
        covmeans, dist_metric = self.weights  # shape(w) = (n_class, n_channels, n_channels)

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


class LDA(ClassifierBlock):
    def __init__(self, n_components=None, tol=1e-5, name="LDA"):
        super(LDA, self).__init__(name)
        self.n_components = n_components
        self.tol = tol
        self.model = sklearn_LDA(solver='svd', n_components=self.n_components, tol=self.tol)

    def fit(self, x, y=None):
        assert len(x.shape) == 2
        assert len(y) == len(x)
        self.model.fit(x, y)
        coef = self.model.coef_
        intercept_ = self.model.intercept_
        if len(intercept_) == 1:
            coef = np.concatenate((-coef, coef), axis=0)/2.
            intercept_ = np.concatenate((-intercept_, intercept_), axis=0)/2.

        w = np.array(coef).astype(np.float64)
        b = np.array(intercept_).astype(np.float64)
        self.weights = [w, b]

    def predict(self, x):
        assert len(x.shape) == 2
        y = self.model.predict_proba(x)
        return y

    def get_keras_layer(self):
        w, b = self.weights

        def LR_predict(_x):
            w_tensor = tf.constant(w.T)  # (n_features, n_class)
            b_tensor = tf.constant(b)  # (n_class,)
            _logits = tf.nn.xw_plus_b(_x, w_tensor, b_tensor)
            _probs = tf.nn.softmax(_logits)
            return _probs

        return Lambda(LR_predict)


class LinearSVC(ClassifierBlock):
    def __init__(self, penalty='l2', loss='squared_hinge', tol=1e-6, C=1.0, class_weight=None,
                 multi_class='ovr', max_iter=1000, name="LinearSVC"):
        super(LinearSVC, self).__init__(name)
        self.penalty = penalty
        self.loss = loss
        self.C = C
        self.class_weight = class_weight
        self.multi_class = multi_class
        self.max_iter = max_iter
        self.tol = tol
        self.model = sklearn_LSVC(penalty=self.penalty, loss=self.loss, tol=self.tol, C=self.C,
                                  multi_class=self.multi_class, class_weight=self.class_weight)

    def fit(self, x, y=None):
        assert len(x.shape) == 2
        assert len(y) == len(x)
        self.model.fit(x, y)
        coef = self.model.coef_
        intercept_ = self.model.intercept_
        if len(intercept_) == 1:
            """
            This is important! When len(class)==2, Sklearn gives a different type of coef_
            and intercept. Look at Sklearn for more information!
            """
            coef = np.concatenate((-coef, coef), axis=0)/2.
            intercept_ = np.concatenate((-intercept_, intercept_), axis=0)/2.
        w = np.array(coef).astype(np.float64)
        b = np.array(intercept_).astype(np.float64)
        self.weights = [w, b]

    def predict(self, x):
        assert len(x.shape) == 2
        y = self.model.decision_function(x)
        if len(y.shape) == 1:
            y = np.reshape(y, newshape=(-1, 1))
            y = np.concatenate((-y/2., y/2.), axis=1)
        y = y - np.max(y, axis=1, keepdims=True)
        y = np.exp(y)
        y_sum = np.sum(y, axis=1, keepdims=True)
        y = y/y_sum
        return y

    def get_keras_layer(self):
        w, b = self.weights  # shape(w) = (n_class, n_features)

        def LR_predict(_x):
            w_tensor = tf.constant(w.T)  # (n_features, n_class)
            b_tensor = tf.constant(b)  # (n_class,)
            _logits = tf.nn.xw_plus_b(_x, w_tensor, b_tensor)
            _probs = tf.nn.softmax(_logits)
            return _probs

        return Lambda(LR_predict)


if __name__ == '__main__':
    # this could be used to see if the keras blocks built correctly.
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    x = 10+10*np.random.rand(10, 250)
    x = x.astype(np.float64)
    # x = np.matmul(x, np.transpose(x, axes=(0, 2, 1)))
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0])

    # model = CSP(n_components=4, transform_into='average_power')
    # model = PCA4Feature(n_components=3, whiten=True)
    # model = ICA(n_components=4)
    # model = Xdawn(n_filters=4, with_xdawn_templates=True, apply_filters=False)
    # model = CovarianceFeature(with_mean_templates=True)
    # model = MDM(dist_metric='riemann')
    # # 'riemann','logeuclid','euclid','logdet'
    # model = TangentSpaceFeature()
    # model = LDA()
    # model = LinearSVC()
    model = Normalizer(norm='l1')
    model.fit(x, y)
    z = model.transform(x)

    input_layer = Input(shape=(250,), dtype=tf.float64)
    features = model.get_keras_layer()(input_layer)
    keras_model = Model(inputs=input_layer, outputs=features)
    z2 = keras_model.predict(x)
    print(np.max(np.abs(z2-z)))