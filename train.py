"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""
import traceback
import numpy as np
import os
from skimage import color, transform, io
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, MaxPooling2D, Conv2D, Flatten
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, History
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator

import random
import numpy as np
import tensorflow as tf
import utils
from utils import create_model, Dataset

seed = 1

# Helper: Early stopping.
early_stopper = EarlyStopping('val_acc', patience=5, verbose=0)


def prec(y_true, y_pred):
    """
      Calculates the precision metrics
    """

    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    TP_FP = K.sum(K.round(K.clip(y_pred, 0, 1)))  # TP + FP == YP
    precision = TP / (TP_FP + K.epsilon())
    return precision


def rcall(y_true, y_pred):
    """
      Calculates the recall metrics
    """

    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    TP_FN = K.sum(K.round(K.clip(y_true, 0, 1)))  # TP + FN == YT
    recall = TP / (TP_FN + K.epsilon())
    return recall


def f1(y_true, y_pred):
    """
      Calculates F1 Score metrics
    """

    p = prec(y_true, y_pred)
    r = rcall(y_true, y_pred)
    return 2*p*r / (r + p)


def compile_model(network, input_shape, out_dim):
    # TODO: Remove loss param
    params = network.params.copy()
    if network.model_type == 'cnn':
        params['cnn_layers'] = network.nb_neurons('cnn')
    params['ann_layers'] = network.nb_neurons('ann')
    optimizer = params['optimizer']
    loss = params['loss']

    model = create_model(params, input_shape, out_dim, network.model_type)
    model.compile(loss=loss, optimizer=optimizer,
                  metrics=['acc', prec])
    return model


def generator(dataset, batch_size=10, flatten=False):
    shape = dataset.img_shape if not flatten else (np.prod(dataset.img_shape),)
    inputs = np.zeros((batch_size, *shape))
    outputs = np.zeros((batch_size, np.prod(dataset.label_shape)))
    while True:
        ids = np.random.choice(dataset.ids, batch_size)
        for i in range(batch_size):
            inputs[i] = dataset.load_image(ids[i], flatten=flatten)
            # flatten always need to be flatten [network output]
            outputs[i] = dataset.load_label(ids[i], flatten=True)
        yield inputs, outputs


def shuffle(x, y):
    seed_ = 1 #random.randint(0, 100)
    state = np.random.get_state()
    np.random.seed(seed_)
    np.random.shuffle(x)
    
    np.random.seed(seed_)
    np.random.shuffle(y)
    np.random.set_state(state)

    return x, y # no returns needed, shuffle is inplace

def generator_v2(X, Y, batch_size=10, input_shape=None, flatten=False):
    l = X.shape[0]    
    inputs = np.zeros((batch_size, *input_shape))
    outputs = np.zeros((batch_size, *Y.shape[1:]))

    shuffle(X, Y)

    while True:
        for i in range( batch_size ):
            idi = random.randint( 0, l-1 )
            
            img = utils.load_image(X[idi], input_shape)
            inputs[i] = img
            outputs[i] = Y[idi]

        # shuffle(inputs, outputs) #no needed

        yield inputs, outputs


def train_and_score(config, network=None, model=None,
                    x_train=None, y_train=None, x_test=None, y_test=None):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating

    """

    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    flatten = config.model_type == 'ann'
    history = History()
    callbacks = [history]
    if config.tb_log_dir:
        callbacks.append(TensorBoard(log_dir=config.tb_log_dir))

    if config.checkpoint:
        check_dir = os.path.join( config.checkpoint, 'w.{epoch:02d}-{val_acc:.2f}.hdf5' )
        checkpoint = ModelCheckpoint( check_dir, monitor='val_acc', save_best_only=True,
                                     save_weights_only=True, period=1)
        callbacks.append(checkpoint)

    try:

        # TODO: Use Dataset class to load labeled data and masked data
        # Temporaly commented

        if not model:
            input_shape = config.input_shape if not flatten \
                else (np.prod(config.input_shape),)
            out_dim = np.prod(config.out_dim)
            model = compile_model(network, input_shape, out_dim)

        if config.early:
            callbacks.append(early_stopper)

        if config.use_generator:
            if x_train is None:
                dataset = Dataset(config.dataset_dir,
                                  img_shape=config.input_shape,
                                  label_shape=config.out_dim,
                                  subset=config.subset)
                dataset.prepare()
                train, val = dataset.split_dataset(0.2)

                fit_generator(model, train, val, config, flatten, callbacks)
                score = model.evaluate_generator(generator(val, 32, flatten),
                                                 steps=10, verbose=0)
            else:
                fit_generator_v2(model, x_train, y_train,
                                 config, flatten, callbacks)
                score = model.evaluate_generator(
                        generator_v2(x_train, y_train,
                        config.batch_size, config.input_shape, flatten),
                        steps=10, verbose=0)

        else:
            # TODO: load_dataset as child of Dataset
            if x_train is None or y_train is None:
                x_train, y_train, x_test, y_test = utils.load_dataset(
                    config, flatten=flatten, split=True)
            fit(model, x_train, y_train, x_test, y_test, config, callbacks)
            if not x_test is None and not y_test is None:
                score = model.evaluate(x_test, y_test, verbose=0)
            else:
                score = model.evaluate(x_train, y_train, verbose=0)
        model.summary()
        K.clear_session()

        return score, history

    except Exception:
        print(traceback.format_exc())
        print('**** Error on fit ****')
        return [0.0, 0.0], None


def fit_generator(model, train, val, config, flatten, callbacks=None):

    model.fit_generator(
        generator(train, config.batch_size, flatten),
        epochs=config.epochs,  # using early stopping, so no real limit
        steps_per_epoch=config.steps_per_epoch,
        validation_data=generator(val, config.batch_size, flatten),
        validation_steps=config.validation_steps,
        callbacks=callbacks,
        verbose=1)


def fit_generator_v2(model, train, val, config, flatten, callbacks=None):

    model.fit_generator(
        generator_v2(train, val, config.batch_size, config.input_shape, flatten),
        epochs=config.epochs,  # using early stopping, so no real limit
        steps_per_epoch=config.steps_per_epoch,
        validation_data=generator_v2(train, val, config.batch_size, config.input_shape, flatten),
        validation_steps=config.validation_steps,
        callbacks=callbacks,
        verbose=1)


def fit(model, x_train, y_train, x_test, y_test, config, callbacks=None):
    """ TODO: change signature to fit_generator signature """
    model.fit(x_train, y_train,
              # callbacks=[history])
              batch_size=config.batch_size,
              epochs=config.epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              #               validation_steps=config.validation_steps,
              callbacks=callbacks)
