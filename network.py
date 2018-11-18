"""Class that represents the network to be evolved."""
import random
import logging
from train import train_and_score
from utils import network_to_json

class Network():
    """Represent a network and let us operate on it.

    Currently only works for an MLP.
    """

    def __init__(self, nn_param_choices=None):
        """Initialize our network.

        Args:
            nn_param_choices (dict): Parameters for the network, includes:
                nb_neurons (list): [64, 128, 256]
                nb_layers (list): [1, 2, 3, 4]
                activation (list): ['relu', 'elu']
                optimizer (list): ['rmsprop', 'adam']
        """
        self.model_type = None
        self.accuracy = 0.
        self.nn_param_choices = nn_param_choices
        self.params = {}  # (dic): represents MLP network parameters

    def create_random(self):
        """Create a random network."""
        for key in self.nn_param_choices:
            self.params[key] = random.choice(self.nn_param_choices[key])

    def create_set(self, params):
        """Set network properties.

        Args:
            network (dict): The network parameters

        """
        self.params = params

    def train(self, config, x_train=None, y_train=None, x_test=None, y_test=None):
        """Train the network and record the accuracy.

        Args:
            config (str): instance of Config.

        """
        self.model_type = config.model_type

        if self.accuracy == 0.:
            score = train_and_score(
                config, self, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
            self.accuracy = score[1]
            print('\n\nACC: %.4f\n\n' % score[1])
            
            if config.verbose == 1:
                self.log( score )
            elif config.verbose == 2 and config.min_acc <= score[1]:
                self.log(score)
            if config.min_acc <= score[1]:
                network_to_json( self )

    def log( self, score ):
        message = '\n{}\nLoss: {} \nAcc: {}\n{}\n\n'.format(
            '='*65, score[0], score[1], '='*65)
        logging.info('\n\nModel:')
        logging.info(self.model())
        logging.info(message)

    def model(self):
        p = self.params
        network = {
            'cnn_nb_layers' : p['cnn_nb_layers' ],
            'cnn_activation': p['cnn_activation'],
            'ann_nb_layers' : p['ann_nb_layers' ],
            'ann_activation': p['ann_activation'],
            'ann_last_activation': p['ann_activation'],
            'optimizer': p['optimizer'],
            'loss': p['loss'],
            'dropout': p['dropout'],
            'pooling': p['pooling'],
            'model_type': self.model_type
        }
        cnn_layers = []
        for i in range(self.params['cnn_nb_layers']):
            cnn_layers.append( self.params['cnn_nb_neurons_%d' % (i+1)] )

        ann_layers = []
        for i in range(self.params['ann_nb_layers']):
            ann_layers.append( self.params['ann_nb_neurons_%d' % (i+1)] )
        
        network['cnn_layers'] = cnn_layers
        network['ann_layers'] = ann_layers

        return network

    def print_network(self):
        """Print out a network."""
        m = self.model()
        str_ = ''
        for key, value in zip( m.keys(), m.values() ):
            str_ += key + ': ' + str(value) + '\n'
        logging.info( str_ )
        logging.info("Network accuracy: %.2f%%\n%s" % (self.accuracy * 100, '='*65))

    def nb_neurons( self, ntype, n_layers=None ):
        if n_layers:
            nb_neurons = [None] * n_layers
        else:
            nb_neurons = [None] * self.params[ntype+'_nb_layers']

        for i in range( len(nb_neurons) ):
            nb_neurons[i] = self.params[ntype+'_nb_neurons_%d'%(i+1)]

        return nb_neurons

