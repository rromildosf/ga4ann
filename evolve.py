import utils
from config import Config
import main
import logging

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO,
    filename='ann_only.txt'
)


class EvConfig(Config):
    # Data settings
    dataset_dir = '../data1_pd_aug5'
    labels_filename = 'labels.txt'
    input_shape = (256, 256, 1)
    out_dim = (2,)

    # GA settings
    model_type = 'cnn'
    generations = 30
    population = 30

    # Network settings
    epochs = 1000  # not exactly
    batch_size = 1
    steps_per_epoch = 10
    validation_steps = 1
    use_generator = True

    #general settings
    min_acc = 0.7
    early = True

    tb_log_dir = 'logs/test_gen'
    checkpoint = 'logs/test_gen'


config = EvConfig()

nn_params = {
    'cnn_nb_layers': [1, 2, 3, 4, 5, 7, 9, 10, 11, 13, 17, 23],
    'cnn_nb_neurons': [2, 32, 12, 64, 128, 13, 17, 23, 29, 31, 43],
    'pooling': [1, 2, 3, 4, 5, 6],  # 2**8
    'cnn_activation': ['relu', 'tanh', 'sigmoid'],

    'ann_nb_layers': [1, 2, 3, 4, 5, 7, 9, 10, 11, 13, 17, 23],
    'ann_nb_neurons': [2, 12, 13, 17, 23, 29, 31, 43, 64],
    'ann_activation': ['relu', 'tanh', 'sigmoid'],
    'ann_last_activation': ['softmax'],  # ['tanh', 'sigmoid', 'softmax'],

    'loss': ['binary_crossentropy'],
    'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad', 'adamax'],
    'dropout': [1, 2, 3, 4, 5, 6],  # 2**8
}

flatten = config.model_type == 'ann'
logging.info("Envolving on dataset: %s" % config.dataset_dir)
x, Y, _ = utils.load_dataset(config, flatten=flatten, split=False, load=False)
main.evolve(config, nn_params, x, Y, None, None, use_cv=True)
