"""
This file is used to build the model pipeline.
"""

from lib.Blocks import ProcessingBlock, ClassifierBlock
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import tensorflow as tf
import numpy as np
import pickle as pickle


class Pipeline(object):
    def __init__(self, processors=[], classifier=None):
        assert len(processors) > 0, 'No processors are used!'
        for processor in processors:
            if not isinstance(processor, ProcessingBlock):
                raise Exception('The processor in \"processors\" must be a ProcessingBlock!')
        if not isinstance(classifier, ClassifierBlock):
            raise Exception('The classifier must be a ClassifierBlock!')

        self.processors = processors
        self.classifier = classifier
        self.fitted = False

    def __fit_processors(self, x, y):
        _x = np.copy(x)
        _y = np.copy(y)
        assert len(_x) == len(_y), "\'x\' and \'y\' should have the same length!"
        for precessor in self.processors:
            precessor.fit(_x, _y)
            _x = precessor.transform(_x)
        return _x, _y

    def __fit_classifier(self, x, y):
        assert len(x) == len(y), "\'x\' and \'y\' should have the same length!"
        self.classifier.fit(x, y)

    def fit(self, x, y):
        assert len(x) == len(y), "\'x\' and \'y\' should have the same length!"
        _x, _y = self.__fit_processors(x, y)
        self.__fit_classifier(_x, _y)
        self.fitted = True

    def predict(self, x):
        _x = np.copy(x)
        for processer in self.processors:
            _x = processer.transform(_x)
        return self.classifier.predict(_x)

    def save(self, save_path):
        weights = []
        for processer in self.processors:
            weights.append(processer.get_weights())
        weights.append(self.classifier.get_weights())
        with open(save_path, 'wb') as f:  # open file with write-mode
            picklestring = pickle.dump(weights, f)  # serialize and save object
            f.flush()
            f.close()

    def load(self, load_path):
        with open(load_path, 'rb') as f:
            weights = pickle.load(f)  # read file and build object
            f.close()
            for i in range(len(self.processors)):
                self.processors[i].load_weights(weights[i])
            self.classifier.load_weights(weights[-1])
            self.fitted = True

    def get_keras_model(self, input_shape):
        assert self.fitted, "The pipeline has not been trained yet!"
        input_x = Input(shape=input_shape, dtype=tf.float64)
        for i in range(len(self.processors)):
            processer = self.processors[i]
            layer = processer.get_keras_layer()
            if i == 0:
                x = layer(input_x)
            else:
                x = layer(x)

        layer = self.classifier.get_keras_layer()
        output = layer(x)
        return Model(inputs=input_x, outputs=output)

    def pipeline_information(self):
        print('processors: {}'.format(', '.join([processor.name for processor in self.processors])))
        print('classifier: {}'.format(self.classifier.name))