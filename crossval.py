import main
from  train import train_and_score
import utils
from config import Config
import numpy as np
from sklearn.model_selection import StratifiedKFold
import logging
import os
import keras.backend as K

def get_models( prefix='acc' ):
    allfiles = os.listdir('./')
    models = [m for m in allfiles if m.startswith(prefix)]
    return models

class CVConfig(Config):
    # Data settings
    dataset_dir = 'data1_pd_aug4'
    labels_filename = 'labels.txt'
    input_shape = (256, 256, 1)
    out_dim = (2,)

    # Network settings
    epochs = 25  #
    use_generator = False

    # GA settings
    model_type = 'ann'
    early = False

    tb_log_dir = './logs/ann_097/'


config = CVConfig()

def apply_crossval( x, y, Y, model, n_folds=10 ):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1)

    c = 0
    log_dir = config.tb_log_dir
    scores = []

    for train, test in skf.split(x, Y):
        print("Running Fold", c+1, "/", n_folds)

        if log_dir:
            config.tb_log_dir = log_dir + '/fold' + str(c)


        keras_model = utils.json_to_model(model, config)
        score = train_and_score(config, model=keras_model,
                              x_train=x[train], y_train=y[train], x_test=x[test], y_test=y[test] )
        K.clear_session() # clear model
        

        print('Acc: %.2f%% \nLoss: %.2f' % (score[1]*100, score[0]))
        scores.append(score[1]*100)

        logging.info('*'*60)
        logging.info(score)
        logging.info('*'*60)
        c += 1
    mean = np.mean(scores)
    std = np.std(scores)
    logging.info('Mean %f' % mean)
    logging.info('Std %f' % std)
    print('Acc: %.2f%% \t Std: %.2f' % (mean, std))

def run_allmodels():
    x, Y, y,  = utils.load_dataset(config, split=False, flatten=True)
    n_folds = 2

    models = get_models()
    for m in models:
        apply_crossval( x, y, Y, model=m, n_folds=n_folds )
        print(m, end=('*'*30))

def run_model():
    x, Y, y,  = utils.load_dataset(config, split=False, flatten=True)
    n_folds = 10

    m = 'model.json'
    apply_crossval( x, y, Y, model=m, n_folds=n_folds )

run_model()