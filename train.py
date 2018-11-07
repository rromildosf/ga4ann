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


import random
import numpy as np
import tensorflow as tf
from utils import create_model

seed = 1
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)


# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)

def split_dataset( inputs, outpts ):
    # Split dataset
    test = int(len(inputs)*0.1/2)
    train = int((len(inputs)-test)/2)
    m = int( len(inputs)/2 )
    y_test = outpts[:test]#.extend(outpts[m:m+test])
    y_test.extend(outpts[m:m+test])
    x_test = inputs[:test]#.extend( inputs[m:m+test] )
    x_test.extend( inputs[m:m+test]  )


    y_train = outpts[:train]#.extend(outpts[m:m+test])
    y_train.extend(outpts[m:m+train])
    x_train = inputs[:train]#.extend( inputs[m:m+test] )
    x_train.extend( inputs[m:m+train]  )

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
        return int( el.split('.')[-2].split('_')[-1] )
  
    Y = np.loadtxt( os.path.join( config.dataset_dir, config.labels_filename ) )
    
    imgs = [ i for i in os.listdir( config.dataset_dir ) if i.endswith(ext) ]
    imgs.sort(key=srt)
  
    X = []
    for i in imgs:
        img = io.imread(  os.path.join( config.dataset_dir, i ), asGray=True )
        X.append( img )
    
    ## used if n_classes=2 for more precise result 50% + 50% clear
    inputs = []
    outpts = []
    for x, y in zip( X, Y ):
        if y == 1.0:
            inputs.append( x )
            outpts.append( y )
    c = 0
    l = len(inputs)
    for x, y in zip(X, Y):
        if y == 0.0 and c < l:
            inputs.append( x )
            outpts.append( y )
            c +=1


    (x_train, y_train), (x_test, y_test) = split_dataset( inputs, outpts )
    x_train = x_train.reshape(x_train.shape[0], *config.input_shape)
    x_test  = x_test.reshape(x_test.shape[0], *config.input_shape)
    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')
    x_train /= 255
    x_test  /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, config.n_classes)
    y_test = to_categorical(y_test, config.n_classes)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    return (x_train, y_train, x_test, y_test)


def compile_model(network, input_shape, nb_classes):
    # TODO: Add compile here
    params = network.params.copy()
    params['ann_layers'] = network.nb_neurons('ann')
    params['cnn_layers'] = network.nb_neurons('cnn')
    
    return create_model( params, input_shape, nb_classes, network.model_type )

def generator(X, Y, batch_size=10,):
    inputs = np.zeros((batch_size, *X.shape[1:]))
    outputs = np.zeros((batch_size, *Y.shape[1:]))
    
    while True:
        for i in range( batch_size ):
            idx = random.randint( 0, X.shape[0]-1 )
            inputs[i] = X[idx]
            outputs[i] = Y[idx]
        yield inputs, outputs


def train_and_score(config, network=None, model=None):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating

    """
    try:
        x_train, y_train, x_test, y_test = load_dataset( config )

        if not model:
            model = compile_model(network, config.input_shape, config.n_classes)
        
        callbacks = [early_stopper]
        if not config.early:
            callbacks = None

        if config.use_generator:
            model.fit_generator( 
                generator( x_train, y_train, config.batch_size ),
                epochs=config.epochs,  # using early stopping, so no real limit
                steps_per_epoch=config.steps_per_epoch,
                validation_data=generator(x_test, y_test, config.batch_size),
                validation_steps=config.validation_steps,
                callbacks=callbacks,
                verbose=1)
        else:
            model.fit(x_train, y_train,
                #callbacks=[history])
                batch_size=config.batch_size,
                epochs=config.epochs,
                verbose=1,
                validation_data=(x_test, y_test),
                callbacks=callbacks)

        score = model.evaluate(x_test, y_test, verbose=0)
        return score, model

    except Exception:
        print( traceback.format_exc() )
        print('**** Error on fit ****')
        return [0.0, 0.0], None
