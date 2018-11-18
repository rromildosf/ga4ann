"""Entry point to evolving the neural network. Start here."""
import logging
from optimizer import Optimizer
from utils import network_to_json, json_to_model
from train import train_and_score

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO,
    filename='log_detect_1.txt'
)


def train_networks(networks, config, x_train=None, y_train=None, x_test=None, y_test=None):
    """Train each network.

    Args:
        networks (list): Current population of networks
        dataset (str): Dataset to use for training/evaluating
    """
    for network in networks:
        network.train(config, x_train=x_train, y_train=y_train,
                      x_test=x_test, y_test=y_test)


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


def generate(generations, population, nn_param_choices, config,
             x_train=None, y_train=None, x_test=None, y_test=None):
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
        train_networks(networks, config, x_train=x_train,
                       y_train=y_train, x_test=x_test, y_test=y_test)

        # Get the average accuracy for this generation.
        average_accuracy = get_average_accuracy(networks)

        # Print out the average accuracy each generation.
        logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
        logging.info('-'*80)

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Do the evolution.
            networks = optimizer.evolve(networks)
        logging.info('Done generation %d' % (i+1))

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


def evolve(config, nn_params, x_train=None, y_train=None, x_test=None, y_test=None):
    """Evolve a network."""
    generations = config.generations  # Number of times to evole the population.
    population = config.population  # Number of networks in each generation.

    if config.model_type == 'cnn':
        cnn_nb_layers = nn_params['cnn_nb_layers']
        cnn_max_layers = 128
        for i in range(cnn_max_layers):
            nn_params['cnn_nb_neurons_%d' % (i+1)] = cnn_nb_layers

    ann_nb_layers = nn_params['ann_nb_layers']
    ann_max_layers = 128
    for i in range(ann_max_layers):
        nn_params['ann_nb_neurons_%d' % (i+1)] = ann_nb_layers

    logging.info("***Evolving %d generations with population %d***" %
                 (generations, population))

    generate(generations, population, nn_params, config,
             x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


class Config():
    # Data settings
    dataset_dir = '../data1_pd_aug4'
    # subset = {'train': 'train_masks'}
    labels_filename = 'labels.txt'
    input_shape = (256, 256, 1)  # must be a tuple
    # tuple of mask shape or tuple of output size. i.e. (10,)
    out_dim = (2,)

    # Network settings
    epochs = 1000  # not exactly
    batch_size = 10
    steps_per_epoch = 100
    validation_steps = 10
    use_generator = False
#     loss = 'binary_crossentropy'

    # GA settings
    model_type = 'cnn'
    generations = 30
    population = 30

    #general settings
    verbose = 1
    min_acc = 0.7
    early = True
    tb_log_dir = '/content/IA/My Drive/carie_seg/tb_logs/aug4/'


def train(config, path,  x_train=None, y_train=None, x_test=None, y_test=None):
    model = json_to_model(path, config)
    score = train_and_score(
        config, model=model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    print(score)
    return score


if __name__ == '__main__':
    # TODO: add Argparser
    config = Config()
    t = 0
    if t == 0:
        nn_params = {
            'cnn_nb_layers': [1, 2, 3, 4, 5, 7, 9, 10, 11, 13, 17, 23],
            'cnn_nb_neurons': [2, 32, 12, 64, 128, 13, 17, 23, 29, 31, 43],
            'cnn_activation': ['relu', 'elu', 'tanh', 'sigmoid'],

            'ann_nb_layers': [1, 2, 3, 4, 5, 7, 9, 10, 11, 13, 17, 23],
            'ann_nb_neurons': [2, 32, 12, 64, 128, 13, 17, 23, 29, 31, 43],
            'ann_activation': ['relu', 'elu', 'tanh', 'sigmoid'],
            'ann_last_activation': ['tanh', 'sigmoid', 'softmax'],
            'loss': ['binary_crossentropy', 'categorical_crossentropy', 'logcosh', 'mean_squared_error'],
            'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad', 'adadelta', 'adamax', 'nadam'],
            'dropout': [1, 2, 3, 4, 5, 6],  # 2**8
            # the great value should be X-2, X is the exponent of size of
            'pooling': [1, 2, 3, 4, 5, 6],
            # image. ie: image shape is (256, 256, deep), 256 is 2**8, so 8 is X, so the great value should be 8-2 = 6
        }
        evolve(config, nn_params)
    else:
        print('Nothing to do ;)')
