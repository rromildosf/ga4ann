"""Entry point to evolving the neural network. Start here."""
import logging
from optimizer import Optimizer
from utils import network_to_json

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO,
    filename='log_aug_1.txt'
)

def train_networks(networks, dataset):
    """Train each network.

    Args:
        networks (list): Current population of networks
        dataset (str): Dataset to use for training/evaluating
    """
    for network in networks:
        network.train(dataset)

def get_average_accuracy(networks):
    """Get the average accuracy for a group of networks.

    Args:
        networks (list): List of networks

    Returns:
        float: The average accuracy of a population of networks.

    """
    total_accuracy = 0
    for network in networks:
        total_accuracy += network.accuracy

    return total_accuracy / len(networks)

def generate(generations, population, nn_param_choices, config):
    """Generate a network with the genetic algorithm.

    Args:
        generations (int): Number of times to evole the population
        population (int): Number of networks in each generation
        nn_param_choices (dict): Parameter choices for networks
        dataset (str): Dataset to use for training/evaluating

    """
    optimizer = Optimizer(nn_param_choices)
    networks = optimizer.create_population(population)

    # Evolve the generation.
    for i in range(generations):

        logging.info("***Doing generation %d of %d***" %
                     (i + 1, generations))

        # Train and get accuracy for networks.
        train_networks(networks, config)

        # Get the average accuracy for this generation.
        average_accuracy = get_average_accuracy(networks)

        # Print out the average accuracy each generation.
        logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
        logging.info('-'*80)

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Do the evolution.
            networks = optimizer.evolve(networks)
        logging.info( 'Done generation %d' % (i+1) )
        
    # Sort our final population.
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)

    # Print out the top 5 networks.
    print_networks(networks[:5])

def print_networks(networks):
    """Print a list of networks.

    Args:
        networks (list): The population of networks

    """
    logging.info('-'*80)
    for network in networks:
        network.print_network()
        network_to_json(network)

def main( config ):
    """Evolve a network."""
    generations = config.generations  # Number of times to evole the population.
    population  = config.population  # Number of networks in each generation.

    nn_param_choices = {
        'cnn_nb_layers' : [1, 2, 3, 4, 5, 7, 9, 10, 11, 13, 17, 23],
        'cnn_nb_neurons': [2, 32, 12, 64, 128, 13, 17, 23, 29, 31, 43],
        'cnn_activation': ['relu', 'elu', 'tanh', 'sigmoid'],
        
        'ann_nb_layers' : [1, 2, 3, 4, 5, 7, 9, 10, 11, 13, 17, 23],
        'ann_nb_neurons': [2, 32, 12, 64, 128, 13, 17, 23, 29, 31, 43],
        'ann_activation': ['relu', 'elu', 'tanh', 'sigmoid'],
        'optimizer'     : ['rmsprop', 'adam', 'sgd', 'adagrad', 'adadelta', 'adamax', 'nadam'],
        'dropout': [1, 2, 3, 4, 5, 6],  
        'pooling' : [1, 2, 3, 4, 5, 6], # can produce erros if cnn_nb_layers is great
    }

    cnn_nb_layers = nn_param_choices['cnn_nb_layers']
    cnn_max_layers = 128
    for i in range( cnn_max_layers ):
        nn_param_choices['cnn_nb_neurons_%d' % (i+1) ] = cnn_nb_layers
    
    ann_nb_layers = nn_param_choices['ann_nb_layers']
    ann_max_layers = 128
    for i in range( ann_max_layers ):
        nn_param_choices['ann_nb_neurons_%d' % (i+1) ] = ann_nb_layers

    logging.info("***Evolving %d generations with population %d***" %
                 (generations, population))

    generate(generations, population, nn_param_choices, config)


class Config():
    # Data settings
    dataset_dir     = '../dataset_padded'
    labels_filename = 'Y_truth.txt'
    input_shape     = (256, 256, 1) # for cnn
    # input_shape     = (256*256*1,)
    n_classes       = 2


    # Network settings
    epochs  = 10000 # not exactly
    steps_per_epoch = 10
    validation_steps = 10
    batch_size = 32
    use_generator = False

    # GA settings
    network_type = 'cnn'
    generations  = 10
    population   = 20



if __name__ == '__main__':
    config = Config()
    main( config )
