import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, MaxPooling2D, Conv2D, Flatten

# TEST -> OK
def network_to_json( network ):
    filename = 'acc[{:.4f}]_opt[{}]_act[{}].json'.format(network.accuracy, 
        network.params['optimizer'],
        network.params['ann_activation'] )
    fp = open( filename, mode='w' )
    json.dump( network.model(), fp, indent=4 )

def create_model(model_params, input_shape, nb_classes, model_type):

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
    model.add(Dense(nb_classes, activation='softmax'))
    
    ## TODO: remove from here
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['acc'])
    return model

def json_to_model( json_path, config ):
    with open( json_path, 'r' ) as fp:
        js = json.load( fp )
    return create_model( js, config.input_shape, config.n_classes, config.model_type )

# test
# network_to_json(None)
