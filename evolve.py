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
    dataset_dir = '../dataset_1_padded'
    labels_filename = 'Y_truth.txt'
    input_shape = (256, 256, 1)
    out_dim = (2,)

    # GA settings
    model_type = 'ann'
    generations = 30
    population = 30
    
    # Network settings
    epochs = 1000  # not exactly
    batch_size = 100
    steps_per_epoch = 100
    validation_steps = 10
    use_generator = False

    #general settings
    min_acc = 0.7
    early = True
    
    tb_log_dir = 'logs/test'
    
config = EvConfig()

nn_params = {
    'ann_nb_layers': [1, 2, 3, 4, 5, 7, 9, 10, 11, 13, 17, 23],
    'ann_nb_neurons': [2, 32, 12, 64, 128, 13, 17, 23, 29, 31, 43],
    'ann_activation': ['relu', 'tanh', 'sigmoid'],
    'ann_last_activation': ['tanh', 'sigmoid', 'softmax'],
    'loss': ['binary_crossentropy', 'categorical_crossentropy', 'logcosh'],
    'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad', 'adadelta', 'adamax', 'nadam'],
    'dropout': [1, 2, 3, 4, 5, 6],  # 2**8
}
        
x_train, y_train, x_test, y_test = utils.load_dataset( config, flatten=True, split=True )
main.evolve( config, nn_params, x_train, y_train, x_test, y_test )