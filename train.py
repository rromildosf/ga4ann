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
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
import keras.backend as K

import random
import numpy as np
import tensorflow as tf
from utils import create_model, Dataset

seed = 1
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)


# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)

# TODO: move to utils


def split_dataset(inputs, outpts):
    # Split dataset
    test = int(len(inputs)*0.1/2)
    train = int((len(inputs)-test)/2)
    m = int(len(inputs)/2)
    y_test = outpts[:test]  # .extend(outpts[m:m+test])
    y_test.extend(outpts[m:m+test])
    x_test = inputs[:test]  # .extend( inputs[m:m+test] )
    x_test.extend(inputs[m:m+test])

    y_train = outpts[:train]  # .extend(outpts[m:m+test])
    y_train.extend(outpts[m:m+train])
    x_train = inputs[:train]  # .extend( inputs[m:m+test] )
    x_train.extend(inputs[m:m+train])

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return (x_train, y_train), (x_test, y_test)


def load_dataset(config, ext='png'):
    """ Load dataset
    returns: (X, Y)
    """
    def srt(el):
        return int(el.split('.')[-2].split('_')[-1])

    Y = np.loadtxt(os.path.join(config.dataset_dir, config.labels_filename))

    imgs = [i for i in os.listdir(config.dataset_dir) if i.endswith(ext)]
    imgs.sort(key=srt)

    X = []
    for i in imgs:
        img = io.imread(os.path.join(config.dataset_dir, i), asGray=True)
        X.append(img)

    # used if n_classes=2 for more precise result 50% + 50% clear
    inputs = []
    outpts = []
    for x, y in zip(X, Y):
        if y == 1.0:
            inputs.append(x)
            outpts.append(y)
    c = 0
    l = len(inputs)
    for x, y in zip(X, Y):
        if y == 0.0 and c < l:
            inputs.append(x)
            outpts.append(y)
            c += 1

    (x_train, y_train), (x_test, y_test) = split_dataset(inputs, outpts)
    x_train = x_train.reshape(x_train.shape[0], *config.input_shape)
    x_test = x_test.reshape(x_test.shape[0], *config.input_shape)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, config.n_classes)
    y_test = to_categorical(y_test, config.n_classes)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    return (x_train, y_train, x_test, y_test)


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


def compile_model(network, input_shape, out_dim, loss):
    # TODO: Add compile here
    params = network.params.copy()
    params['ann_layers'] = network.nb_neurons('ann')
    params['cnn_layers'] = network.nb_neurons('cnn')
    optimizer = params['optimizer']

    model = create_model(params, input_shape, out_dim, network.model_type)
    model.compile(loss=loss, optimizer=optimizer, metrics=['acc', prec, rcall, f1])
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


def train_and_score(config, network=None, model=None):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating

    """
    try:
        # Temporaly commented
        # x_train, y_train, x_test, y_test = load_dataset( config )
        dataset = Dataset(config.dataset_dir,
                          img_shape=config.input_shape,
                          label_shape=config.out_dim,
                          subset=config.subset)
        dataset.prepare()
        train, val = dataset.split_dataset(0.2)

        flatten = config.model_type == 'ann'

        if not model:
            input_shape = config.input_shape if not flatten \
                else (np.prod(config.input_shape),)
            out_dim = np.prod(config.out_dim)
            model = compile_model(network, input_shape, out_dim, config.loss)

        callbacks = [early_stopper]
        if not config.early:
            callbacks = None

        if config.use_generator:
            fit_generator(model, train, val, config, flatten, callbacks)
            score = model.evaluate_generator(generator(val, 32, flatten),
                                             steps=10, verbose=0)

        else:
            # TODO: load_dataset as child of Dataset
            x_train, y_train, x_test, y_test = load_dataset(config)
            fit(model, x_train, y_train, x_test, y_test, config, callbacks)
            score = model.evaluate(x_test, y_test, verbose=0)

        return score, model

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


def fit(model, x_train, y_train, x_test, y_test, config, callbacks=None):
    """ TODO: change signature to fit_generator signature """
    model.fit(x_train, y_train,
              # callbacks=[history])
              batch_size=config.batch_size,
              epochs=config.epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=callbacks)
