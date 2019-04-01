# EEGAdversary
This is a toolbox to construct adversarial examples of EEG signals. The traditional EEG extraction methods and classifiers are re-implemented in Tensorflow.

## 1 Requirements

Tensorflow = 1.13.1 [https://tensorflow.google.cn/](https://tensorflow.google.cn/ "https://tensorflow.google.cn/")

pyriemann = 0.2.5 [https://pyriemann.readthedocs.io/en/latest/](https://pyriemann.readthedocs.io/en/latest/ "https://pyriemann.readthedocs.io/en/latest/")

mne = 0.17.1 [http://www.martinos.org/mne/stable/index.html](http://www.martinos.org/mne/stable/index.html "http://www.martinos.org/mne/stable/index.html")


## 2 How to use it

### 2.1 Blocks

Sorts of classic methods used in EEG have been re-implemented in Tensorflow in our library as you could seen in *lib/Blocks.py*. 

The blocks including:

**Processing Blocks**: xDAWN, CSP, ICA, PCA, covariance feature, tangent space feature ('riemann', 'logdet', 'logeuclid', 'euclid')

**Classifiers**: logistic regression, SVM, LDA, MDM ('riemann', 'logdet', 'logeuclid', 'euclid')

### 2.2 Pipeline

Here is a small example for how to build the precessing pipeline. Let's assume you want to build a pipeline including (xDAWN, Covariance, tangent space feature ('riemann'), logistic regression) for P300 classification. Then, you could simply use

    from lib import Blocks
    from lib import Pipeline
    
    processers = [
	    Blocks.Xdawn(n_filters=8, with_xdawn_templates=True, apply_filters=True),
	    Blocks.CovarianceFeature(with_mean_templates=False),
	    Blocks.TangentSpaceFeature(mean_metric='riemann', name='TangentSpace({})'.format('riemann'))
    ]
    
    classifier = Blocks.LogisticRegression(class_weight=class_weight)
    
    model = Pipeline(processers, classifier)

to build it. And if you want to **see the information of the pipeline**:

	model.pipeline_information()

Then if you want to **train the pipeline**:
	
	model.fit(epochs, y)

if you want to **get the prediction of the pipeline**:

	model.predict(epochs)

when you want to **get the model re-implemented in Tensorflow**:

	keras_model = model.get_keras_model(input_shape=(n_channel, length))

when you want to **save the model**:
	
	model.save(save_path)

and of course **load the model** (Please be aware that now only the tensorflow version could be used after being loaded):

	model.load(load_path)

### 2.3 Construct Adversarial Examples

Please look at *lib/KerasAdversary.py* for more information. This file mainly contains several classes for adversarial attacks.

1. **WhiteBoxAttacks**: L-BFGS-B, FGSM, FGM, BIM, C&W(TODO)
2. **BlackBoxAttacks**: TODO

Now it supports both target attacks and non-target attacks.

Adversarial attack can be quite simple with our library. For example, if *model* is the target model built with Keras, then it can be attacked by FGSM with the following commands:
	
	from lib.KerasAdversary import WhiteBoxAttacks
	import tensorflow.keras.backen as K	
	AttackAgent = WhiteBoxAttacks(model, K.get_session())
	adv_x = AttackAgent.fgsm(x, y, epsilon=0.1)

where *adv_x* are adversarial examples of x.

**NOTE**: If you want to perform **target attacks**, then you should set the parameter **target=True** and **y is your target label**. For example,

	from lib.KerasAdversary import WhiteBoxAttacks
	import tensorflow.keras.backen as K	
	AttackAgent = WhiteBoxAttacks(model, K.get_session())
	adv_x = AttackAgent.fgsm(x, y, target=True, epsilon=0.1)

# About Our Paper

We give out all the code to reproduce the results in our paper. Please check our paper and files for more information.

**Related files**:

- train.py
- test.py
- generate_adversarial_templates.py
- target_attack_with_templates.py
- get_averaged_epochs.py

**For visualization**:

- plot_attack_result.py
- plot_EEG_signal.py
- matlab/plot_topomaps_together.m (you should run get_averaged_epochs.py first).
