import os
import json
import numpy as np

import keras.backend as K
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Dropout, MaxPooling2D, Conv2D, Flatten, BatchNormalization
from keras.utils.np_utils import to_categorical
from keras.initializers import glorot_normal
from keras import regularizers

from skimage import io
from sklearn.model_selection import StratifiedKFold


def split_dataset(x, y, n_splits=10, out_dim=None, fold=0):
    fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    i = 0
    for train, test in fold.split(x, y):
        if i == fold:
            if out_dim:
                return x[train], to_categorical(y[train]), \
                    x[test], to_categorical(y[test])
            return x[train], y[train], x[test], y[test]


def load_image(path, shape=None):
    img = io.imread(path, asGray=True)
    img = img.astype(np.float32)
    img /= 255.
    if shape:
        img = img.reshape(*shape)
    return img


def load_dataset(config, ext='png', flatten=False, split=False, fold=0, load=True):
    """ Load dataset
    returns: (X, Y)
    """
    def srt(el):
        return int(el.split('.')[-2].split('_')[-1])

    Y = np.loadtxt(os.path.join(config.dataset_dir, config.labels_filename))

    imgs = [i for i in os.listdir(config.dataset_dir) if i.endswith(ext)]
    imgs.sort(key=srt)

    X = []
    if load:
        for i in imgs:
            img = io.imread(os.path.join(config.dataset_dir, i), asGray=True)
            X.append(img)
    else:
        X = [os.path.join(config.dataset_dir, i) for i in imgs]

    # 50% + 50%
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
    X = None
    # convert class vectors to binary class matrices
    out_dim = np.prod(config.out_dim)
    inputs = np.array(inputs)
    outpts = np.array(outpts)

    if load:
        input_shape = config.input_shape if not flatten else (
            np.prod(config.input_shape),)
        inputs = inputs.reshape(inputs.shape[0], *input_shape)
        inputs = inputs.astype(np.float32)
        inputs /= 255.

    if split:
        return split_dataset(inputs, outpts, out_dim=out_dim, fold=fold)
    # changed to use cross validation
    return inputs, outpts, to_categorical(outpts, out_dim)


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


# TEST -> OK
def network_to_json(network):
    filename = 'acc[{:.4f}]_opt[{}]_act[{}].json'.format(network.accuracy,
                                                         network.params['optimizer'],
                                                         network.params['ann_activation'])
    fp = open(filename, mode='w')
    json.dump(network.model(), fp, indent=4)


def create_model(model_params, input_shape, out_dim, model_type):
    if model_type == 'ann':
        input_shape = (np.prod(input_shape),)
    # Get our network parameters.
    if model_type == 'cnn':
        cnn_nb_layers = model_params['cnn_nb_layers']
        cnn_nb_neurons = model_params['cnn_layers']
        cnn_activation = model_params['cnn_activation']
        pooling = model_params['pooling']

    ann_nb_layers = model_params['ann_nb_layers']
    ann_nb_neurons = model_params['ann_layers']
    ann_activation = model_params['ann_activation']
    ann_last_activation = model_params['ann_last_activation']

    dropout = model_params['dropout']

    model = Sequential()
    index = 0  # start index of ann
    init = glorot_normal(1)
    if model_type == 'cnn':
        model.add(Conv2D(cnn_nb_neurons[0], (3, 3), padding='same',
                         activation=cnn_activation,
                         input_shape=input_shape))

        pool = 0
        for i in range(1, cnn_nb_layers):
            model.add(
                Conv2D(cnn_nb_neurons[i], (3, 3), activation=cnn_activation, padding='same'))
            if i % 2 != 0 and pool <= pooling:
                model.add(MaxPooling2D(2))
                pool += 1

        model.add(Flatten())
    else:  # mlp
        # Add each layer.
        index = 1
        model.add(Dense(ann_nb_neurons[0], activation=ann_activation,
                        # input_shape=input_shape,
                        # kernel_initializer=init,
                        bias_initializer='zeros'))
    drop = 0
    for i in range(index, ann_nb_layers):
        model.add(Dense(ann_nb_neurons[i], activation=ann_activation,
                        # kernel_initializer=init,
                        # bias_initializer='zeros',
                        kernel_regularizer=regularizers.l2(0.01),
                        activity_regularizer=regularizers.l1(0.01)))

        if i % 2 != 0 and drop <= dropout:
            model.add(Dropout(0.5))
            drop += 1
        if i % 3 == 0:
            model.add(BatchNormalization())
    model.add(BatchNormalization())

    # Output layer.
    model.add(Dense(out_dim, activation=ann_last_activation,
                    # kernel_initializer=init,
                    # bias_initializer='zeros'
                    ))

    return model


def json_to_model(json_path, config):
    with open(json_path, 'r') as fp:
        js = json.load(fp)
    dim = np.prod(config.out_dim)
    model = create_model(js, config.input_shape, dim, js['model_type'])
    model.compile(loss=js['loss'], optimizer=js['optimizer'],
                  metrics=['acc', prec, rcall, f1])
    return model


class Dataset():

    def __init__(self, dataset_dir=None, img_shape=None, label_shape=None, subset=None):
        self.img_shape = img_shape
        self.label_shape = label_shape
        self.dataset_dir = dataset_dir
        self.subset = subset

        if dataset_dir:
            self.prepare()

    def prepare(self, dataset_dir=None, subset=None, validation_split=0.0):
        assert self.dataset_dir or dataset_dir and self.subset or subset, 'Necessary dataset_dir and subset'
        self.validation_split = validation_split
        if dataset_dir:
            self.dataset_dir = dataset_dir
        if subset:
            self.subset = subset

        # TODO: rewrite to use first element
        x_dir = None
        y_dir = None
        for x, y in zip(self.subset.keys(), self.subset.values()):
            x_dir = os.path.join(self.dataset_dir, x)
            y_dir = os.path.join(self.dataset_dir, y)

        img_names = sorted(os.listdir(x_dir))
        label_names = sorted(os.listdir(y_dir))

        self.img_paths = []
        self.label_paths = []
        for i, m in zip(img_names, label_names):
            if i.split('.')[-0] != m.split('.')[0]:
                continue
            self.img_paths.append(os.path.join(x_dir, i))
            self.label_paths.append(os.path.join(y_dir, m))
        self.ids = [i for i in range(len(self.img_paths))]

    def __load(self, arr, idx, shape, flatten=False):
        mode = 'rgb'
        if shape[-1] != 3:
            mode = 'grayscale'
        img = np.array(image.load_img(arr[idx],
                                      color_mode=mode, target_size=shape))
        if shape[-1] != 3:
            img = img.reshape(*img.shape, 1)
        if flatten:
            img = img.reshape(np.prod(img.shape))
        return img

    def load_image(self, idx, flatten=False):
        return self.__load(self.img_paths, idx, self.img_shape, flatten)

    def load_label(self, idx, flatten=False):
        return self.__load(self.label_paths, idx, self.label_shape, flatten)

    def load_set(self, ids, flatten=False):
        i_shape = (np.prod(self.img_shape),) if flatten else self.img_shape
        l_shape = (np.prod(self.label_shape),) if flatten else self.label_shape

        imgs = np.zeros((len(ids), *i_shape))
        labels = np.zeros((len(ids), *l_shape))

        for i in range(len(ids)):
            imgs[i] = self.load_image(i, flatten)
            labels[i] = self.load_label(i, flatten)
        return (imgs, labels)

    def split_dataset(self, validation_split=0.2, suffle=True, seed=None):
        # TODO: implement suffle and seed functionalities
        # TODO: shorten this method

        train_set = Dataset(img_shape=self.img_shape,
                            label_shape=self.label_shape)
        val_set = Dataset(img_shape=self.img_shape,
                          label_shape=self.label_shape)

        size = len(self.ids)
        val_size = int(size*validation_split)
        train_size = size - val_size

        train_ids = range(train_size)
        val_ids = range(val_size)

        train_img_paths = self.img_paths[:train_size]
        train_label_paths = self.label_paths[:train_size]

        val_img_paths = self.img_paths[train_size:]
        val_label_paths = self.label_paths[train_size:]

        train_set.ids = train_ids
        train_set.img_paths = train_img_paths
        train_set.label_paths = train_label_paths

        val_set.ids = val_ids
        val_set.img_paths = val_img_paths
        val_set.label_paths = val_label_paths

        return train_set, val_set
# test
# network_to_json(None)
# data = Dataset('../../dataset_aug', subset={'train': 'train_masks'},
#     img_shape=(256, 256), label_shape=(64, 64))
# data.prepare()
# train, val = data.split_dataset()
# print( train.load_set( train.ids[:10] , as_gray=True, flatten=False)[1].shape )
