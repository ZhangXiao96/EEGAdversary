"""
This file is used to perform some adversarial attack methods on the deep learning models built with Keras.
Now involve:
    white-box attack:
        FGM, L-BFGS-B, FGSM, BIM, C&W.
    black_box attack:

We will update the methods irregularly.
Please email to xiao_zhang@hust.edu.cn if you have any questions.
"""
from tensorflow.keras import backend as K
from lib import utils
from tqdm import tqdm
import numpy as np
from scipy import optimize

class WhiteBoxAttacks(object):
    """
    This class provides a simple interface to perform white-box attacks (both target and none-target) on keras models.
    For example, if you want to perform FGSM, you can simply use

        AttackAgent = WhiteBoxAttacks(target_model, session)
        adv_x = AttackAgent.fgsm(x, y, target=False, epsilon=0.1)

    to generate adversarial examples of x.
    """

    def __init__(self, model, sess, loss_fn=None):
        """
        To generate the White-box Attack Agent.
        RNN is not supported now.
        :param model: the target model which should have the input tensor, the target tensor and the loss tensor.
        :param sess: the tensorflow session.
        :param loss_fn: None if using original loss of the model.
               You can also use your own loss function instead, e.g. keras.losses.sparse_categorical_crossentropy
               NOTE: the original loss always involves regular loss!
        """
        self.model = model
        self.input_tensor = model.inputs[0]
        self.output_tensor = model.outputs[0]
        self.target_tensor = model.targets[0]
        self._sample_weights = model.sample_weights[0]
        if loss_fn is None:
            self.loss_tensor = model.total_loss
            self.gradient_tensor = K.gradients(self.loss_tensor, self.input_tensor)[0]
        else:
            self.set_loss_function(loss_fn)
        self.sess = sess

    def set_loss_function(self, loss_fn):
        score = loss_fn(self.target_tensor, self.output_tensor)
        ndim = K.ndim(score)
        weight_ndim = K.ndim(self._sample_weights)
        score = K.mean(score, axis=list(range(weight_ndim, ndim)))
        score *= self._sample_weights
        score /= K.mean(K.cast(K.not_equal(self._sample_weights, 0), K.floatx()))
        self.loss_tensor = K.mean(score)
        self.gradient_tensor = K.gradients(self.loss_tensor, self.input_tensor)[0]

    def get_model(self):
        return self.model

    def get_sess(self):
        return self.sess

    def _get_batch_loss(self, x_batch, y_batch, sample_weights=None, mean=True):
        num = len(y_batch)
        y_batch = np.reshape(y_batch, newshape=[num, 1])
        if sample_weights is None:
            sample_weights = np.ones((num,))
        feed_dict = {
            self.input_tensor: x_batch,
            self.target_tensor: y_batch,
            self._sample_weights: sample_weights,
            K.learning_phase(): 0
        }
        batch_loss = self.sess.run(self.loss_tensor, feed_dict=feed_dict)
        if not mean:
            batch_loss = num * batch_loss
        return batch_loss

    def _get_batch_gradients(self, x_batch, y_batch, sample_weights=None):
        num = len(y_batch)
        y_batch = np.reshape(y_batch, newshape=[num, 1])
        if sample_weights is None:
            sample_weights = np.ones((num,))
        feed_dict = {
            self.input_tensor: x_batch,
            self.target_tensor: y_batch,
            self._sample_weights: sample_weights,
            K.learning_phase(): 0
        }
        gradient_batch = self.sess.run(self.gradient_tensor, feed_dict=feed_dict)
        gradient_batch = num * gradient_batch  # To remove 1/Batchsize before the loss
        return gradient_batch

    def get_gradients(self, x, y, batch_size=256):
        """
        This function is used to get the gradients \Delta_{x}Loss(x,y;\theta)
        :param x: the normal examples
        :param y: the labels of x
        :param batch_size: batch size
        :return: gradients
        """
        gradients = []
        data = zip(x, y)
        batches = list(utils.batch_iter(data, batchsize=batch_size, shuffle=False))
        for batch in tqdm(batches):
            x_batch, y_batch = zip(*batch)
            gradient_batch = self._get_batch_gradients(x_batch=x_batch, y_batch=y_batch)
            gradients.append(gradient_batch)
        gradients = np.concatenate(gradients, axis=0)
        return gradients

    def l_bfgs_b(self, x, y, batch_size=256, initial_const=1e10, max_iteration=10, binary_search_steps=20, clip_min=None, clip_max=None):
        """
        L-BFGS-B (our implementation referenced CleverHans https://github.com/tensorflow/cleverhans)
        The original paper can be found at: https://arxiv.org/abs/1412.6572
        @Article{LBFGSB,
            author        = {Christian Szegedy and Wojciech Zaremba and Ilya Sutskever and Joan Bruna and Dumitru Erhan and Ian J. Goodfellow and Rob Fergus},
            title         = {Intriguing properties of neural networks},
            journal       = {CoRR},
            year          = {2013},
            volume        = {abs/1312.6199},
            archiveprefix = {arXiv},
            url           = {http://arxiv.org/abs/1312.6199},
        }
        :param x: the normal examples
        :param y: target labels of x
        :param batch_size: batch size
        :param initial_const: initial constant for
        :param max_iteration: max iterations for L-BFGS-B
        :param clip_min: minimum input component value. If `None`, clipping is not performed on lower
        interval edge.
        :param clip_max: maximum input component value. If `None`, clipping is not performed on upper
        interval edge.
        :return: adversarial examples of x.
        """
        if clip_min is None:
            clip_min = -np.Inf
        if clip_max is None:
            clip_max = np.Inf

        def objective(batch_adv_x, batch_y, batch_x, const):
            batch_adv_x = np.reshape(batch_adv_x, newshape=batch_x.shape)

            class_loss = self._get_batch_loss(x_batch=batch_adv_x, y_batch=batch_y, sample_weights=const, mean=False)
            constrain_loss = np.sum(np.square(batch_adv_x - batch_x))
            class_gradients = self._get_batch_gradients(batch_adv_x, batch_y, sample_weights=const)
            constrain_gradients = 2 * np.reshape(batch_adv_x - batch_x, newshape=[len(batch_x), -1])

            loss = class_loss + constrain_loss
            gradients = class_gradients.flatten().astype(float) + constrain_gradients.flatten().astype(float)
            return loss, gradients

        data = zip(x, y)
        batches = list(utils.batch_iter(data, batchsize=batch_size, shuffle=False))
        adv_x = []

        for batch in tqdm(batches):
            x_batch, y_batch = zip(*batch)
            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)
            num = len(y_batch)
            CONST = np.ones([num, ]) * initial_const

            min_x_bound = np.ones(x_batch.shape[:]) * clip_min
            max_x_bound = np.ones(x_batch.shape[:]) * clip_max
            clip_bound = list(zip(min_x_bound.flatten(), max_x_bound.flatten()))

            # set the lower and upper bounds accordingly
            lower_bound = np.zeros([num, ])
            upper_bound = np.ones([num, ]) * 1e10

            o_bestl2 = [1e10] * num
            o_bestattack = np.copy(x_batch)

            for step in range(binary_search_steps):
                # The last iteration (if we run many steps) repeat the search once.
                if step == binary_search_steps - 1:
                    CONST = upper_bound
                adv_x_batch, min_loss, _ = optimize.fmin_l_bfgs_b(
                    objective, x_batch.flatten().astype(float),
                    args=(y_batch, x_batch, CONST),
                    bounds=clip_bound,
                    maxiter=max_iteration,
                    iprint=0
                )
                adv_x_batch = np.reshape(adv_x_batch, newshape=x_batch.shape)
                assert np.amax(adv_x_batch) <= clip_max and \
                       np.amin(adv_x_batch) >= clip_min, \
                    'fmin_l_bfgs_b returns are invalid'

                preds = np.argmax(self.model.predict(adv_x_batch, verbose=0), axis=1)

                l2s = np.zeros(num)
                for i in range(num):
                    l2s[i] = np.sum(np.square(adv_x_batch[i] - x_batch[i]))

                for e, (l2, pred, ii) in enumerate(zip(l2s, preds, adv_x_batch)):
                    if l2 < o_bestl2[e] and preds[e] == y_batch[e]:
                        o_bestl2[e] = l2
                        o_bestattack[e] = ii

                # adjust the constant as needed
                for e in range(num):
                    if preds[e] == y_batch[e]:
                        # success, divide const by two
                        upper_bound[e] = min(upper_bound[e], CONST[e])
                        if upper_bound[e] < 1e9:
                            CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                    else:
                        # failure, either multiply by 10 if no solution found yet
                        #          or do binary search with the known upper bound
                        lower_bound[e] = max(lower_bound[e], CONST[e])
                        if upper_bound[e] < 1e9:
                            CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                        else:
                            CONST[e] *= 10
                o_bestl2 = np.array(o_bestl2)

            adv_x.append(np.array(o_bestattack))
        adv_x = np.concatenate(adv_x, axis=0)
        return adv_x

    def fgm(self, x, y, target=False, epsilon=0.1, norm_ord=None, batch_size=256, clip_min=None, clip_max=None, tol=1e-8):
        """
        Fast Gradient Method (fgm).
        Just add the gradients whose ord norm is epsilon (fixed).
        :param x: the normal examples
        :param y: the labels of x for target attack or none-target attack (according to target=True or False)
        :param target: True -> target attack and y is the target. False -> none-target attack and y is the true label.
        :param epsilon: the limit of the norm of the gradient.
        :param norm_ord: the ord of the norm. If is None, the gradients will not be normalized.
        :param batch_size: batch size
        :param clip_min: minimum input component value. If `None`, clipping is not performed on lower
        interval edge.
        :param clip_max: maximum input component value. If `None`, clipping is not performed on upper
        interval edge.
        :return: adversarial examples of x.
        """
        gradients = self.get_gradients(x, y, batch_size=batch_size)

        if norm_ord is not None:
            adv_flat = np.reshape(gradients, newshape=[gradients.shape[0], -1])
            norms = np.linalg.norm(adv_flat, ord=norm_ord, axis=1, keepdims=True) + tol
            gradients = np.reshape(adv_flat / norms, newshape=gradients.shape)

        adv_noise = epsilon * gradients
        if target:
            adv_x = x - adv_noise
        else:
            adv_x = x + adv_noise

        if (clip_min is not None) or (clip_max is not None):
            adv_x = np.clip(adv_x, a_min=clip_min, a_max=clip_max)
        return adv_x

    def fgsm(self, x, y, target=False, epsilon=0.1, batch_size=256, clip_min=None, clip_max=None):
        """
        Fast Gradient Sign Method (FGSM).
        The original paper can be found at: https://arxiv.org/abs/1412.6572
        @Article{FGSM,
          author        = {Ian J. Goodfellow and Jonathon Shlens and Christian Szegedy},
          title         = {Explaining and Harnessing Adversarial Examples},
          journal       = {CoRR},
          year          = {2014},
          volume        = {abs/1412.6572},
          archiveprefix = {arXiv},
          eprint        = {1412.6572},
          url           = {http://arxiv.org/abs/1412.6572},
        }
        :param x: the normal examples
        :param y: the labels of x for target attack or none-target attack (according to target=True or False)
        :param target: True -> target attack and y is the target. False -> none-target attack and y is the true label.
        :param epsilon: the limit of the permutation
        :param batch_size: batch size
        :param clip_min: minimum input component value. If `None`, clipping is not performed on lower
        interval edge.
        :param clip_max: maximum input component value. If `None`, clipping is not performed on upper
        interval edge.
        :return: adversarial examples of x.
        """
        gradients = self.get_gradients(x, y, batch_size=batch_size)
        adv_noise = epsilon * np.sign(gradients)
        if target:
            adv_x = x - adv_noise
        else:
            adv_x = x + adv_noise

        if (clip_min is not None) or (clip_max is not None):
            adv_x = np.clip(adv_x, a_min=clip_min, a_max=clip_max)
        return adv_x

    def bim(self, x, y, target=False, epsilon=0.1, iterations=3, batch_size=256, clip_min=None, clip_max=None):
        """
        Basic Iterative Method (BIM).
        The original paper can be found at: https://arxiv.org/abs/1607.02533
        @Article{BIM,
          author        = {Alexey Kurakin and Ian J. Goodfellow and Samy Bengio},
          title         = {Adversarial examples in the physical world},
          journal       = {CoRR},
          year          = {2016},
          volume        = {abs/1607.02533},
          archiveprefix = {arXiv},
          eprint        = {1607.02533},
          url           = {http://arxiv.org/abs/1607.02533},
        }
        :param x: the normal examples
        :param y: the labels of x for target attack or none-target attack (according to target=True or False)
        :param target: True -> target attack and y is the target. False -> none-target attack and y is the true label.
        :param epsilon: the limit of the permutation
        :param iterations: number of attack iterations.
        :param batch_size: batch size
        :param clip_min: minimum input component value. If `None`, clipping is not performed on lower
        interval edge.
        :param clip_max: maximum input component value. If `None`, clipping is not performed on upper
        interval edge.
        :return: adversarial examples of x.
        """
        adv_x = x
        for iteration in range(iterations):
            print('Performing BIM: {}/{} iterations'.format(iteration+1, iterations))
            adv_x = self.fgsm(adv_x, y, target=target, epsilon=epsilon, batch_size=batch_size, clip_min=clip_min, clip_max=clip_max)
        return adv_x

    def carlini_and_wagner(self, x, y):
        """
        Carlini & Wagner (C&W).
        The original paper can be found at: https://arxiv.org/abs/1608.04644
        @Article{CandW,
          author  = {Nicholas Carlini and David A. Wagner},
          title   = {Towards Evaluating the Robustness of Neural Networks},
          journal = {CoRR},
          year    = {2016},
          volume  = {abs/1608.04644},
          url     = {https://arxiv.org/abs/1608.04644},
        }
        TODO:
        """
        pass
