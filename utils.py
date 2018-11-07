import json
import os
import numpy as np
from skimage.io import imread
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Dropout, MaxPooling2D, Conv2D, Flatten

# TEST -> OK
def network_to_json( network ):
    filename = 'acc[{:.4f}]_opt[{}]_act[{}].json'.format(network.accuracy, 
        network.params['optimizer'],
        network.params['ann_activation'] )
    fp = open( filename, mode='w' )
    json.dump( network.model(), fp, indent=4 )

def create_model(model_params, input_shape, out_dim, model_type):

    # Get our network parameters.
    if model_type == 'cnn':
        cnn_nb_layers  = model_params['cnn_nb_layers']
        cnn_nb_neurons = model_params['cnn_layers']
        cnn_activation = model_params['cnn_activation']
        pooling   = model_params['pooling']

    ann_nb_layers  = model_params['ann_nb_layers']
    ann_nb_neurons = model_params['ann_layers']
    ann_activation = model_params['ann_activation']

    optimizer = model_params['optimizer']
    dropout   = model_params['dropout']

    model = Sequential()
    index = 0  # start index of ann

    if model_type == 'cnn':
        model.add(Conv2D(cnn_nb_neurons[0], (3, 3), padding='same',
                         activation=cnn_activation,
                         input_shape=input_shape))
            
        pool = 0
        for i in range(1, cnn_nb_layers):
            model.add(
                Conv2D(cnn_nb_neurons[i], (3, 3), activation=cnn_activation, padding='same'))
            if i % pooling == 0 and pool <= pooling:
                model.add(MaxPooling2D(2))
                pool += 1

        model.add(Flatten())
    else:  # mlp
        # Add each layer.
        index = 1
        model.add(Dense(ann_nb_neurons[0], activation=ann_activation,
                        input_shape=input_shape))
    drop = 0
    for i in range(index, ann_nb_layers):
        model.add(Dense(ann_nb_neurons[i], activation=ann_activation))
        if i % dropout == 0 and drop <= dropout:
            model.add(Dropout(0.5))
            drop += 1

    # Output layer.
    model.add(Dense(out_dim, activation='softmax'))
    
    ## TODO: remove from here
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['acc'])
    return model

def json_to_model( json_path, config ):
    with open( json_path, 'r' ) as fp:
        js = json.load( fp )
    return create_model( js, config.input_shape, config.n_classes, config.model_type )


class Dataset():
    
    def __init__(self, dataset_dir=None, img_shape=None, label_shape=None, subset=None):
        self.img_shape   = img_shape
        self.label_shape = label_shape
        self.dataset_dir = dataset_dir
        self.subset      = subset

        if dataset_dir:
            self.prepare()
    
    def prepare(self, dataset_dir=None, subset=None, validation_split=0.0):
        assert self.dataset_dir or dataset_dir and self.subset or subset, 'Necessary dataset_dir and subset'
        self.validation_split = validation_split
        if dataset_dir: self.dataset_dir = dataset_dir
        if subset: self.subset = subset
        
        # TODO: rewrite to use first element
        x_dir = None
        y_dir = None
        for x, y in zip(self.subset.keys(), self.subset.values()):
            x_dir = os.path.join( self.dataset_dir, x )
            y_dir = os.path.join( self.dataset_dir, y )
        
        img_names = sorted(os.listdir(x_dir))
        label_names = sorted(os.listdir(y_dir))
        
        self.img_paths = []
        self.label_paths = []
        for i, m in zip( img_names, label_names ):
            if i.split('.')[-0] != m.split('.')[0]:
                continue
            self.img_paths.append( os.path.join(x_dir, i) )
            self.label_paths.append( os.path.join(y_dir, m) )
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
            img = img.reshape(np.prod(img.shape) )
        return img

    def load_image(self, idx, flatten=False):
        return self.__load( self.img_paths, idx, self.img_shape, flatten)
    
    def load_label(self, idx, flatten=False):
        return self.__load(self.label_paths, idx, self.label_shape, flatten)

    def load_set(self, ids, flatten=False):
        i_shape = (np.prod(self.img_shape),)   if flatten else self.img_shape
        l_shape = (np.prod(self.label_shape),) if flatten else self.label_shape
        
        imgs = np.zeros( (len(ids), *i_shape) )
        labels = np.zeros( (len(ids), *l_shape) )

        for i in range(len(ids)):
            imgs[i] = self.load_image( i, flatten )
            labels[i] = self.load_label( i, flatten )
        return (imgs, labels)
    
    def split_dataset(self, validation_split=0.2, suffle=True, seed=None):
        # TODO: implement suffle and seed functionalities
        # TODO: shorten this method

        train_set = Dataset(img_shape=self.img_shape, label_shape=self.label_shape)
        val_set  = Dataset(img_shape=self.img_shape, label_shape=self.label_shape)

        size = len(self.ids)
        val_size = int(size*validation_split)
        train_size = size - val_size

        train_ids = range(train_size)
        val_ids   = range(val_size)

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

