import main
import utils
from config import Config
import numpy as np
from sklearn.model_selection import StratifiedKFold
import logging

class CVConfig(Config):
    # Data settings
    dataset_dir = '../dataset_1_padded'
    labels_filename = 'Y_truth.txt'
    input_shape = (256, 256, 1)
    out_dim = (2,)

    # Network settings
    epochs = 150  #
    use_generator = False

    # GA settings
    model_type = 'ann'
    early = False

    tb_log_dir = None #'/content/IA/My Drive/IA/carie_class/logs/test_del'


config = CVConfig()

x, Y, y,  = utils.load_dataset(config, split=False, flatten=True)
n_folds = 10
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1)

c = 0
log_dir = config.tb_log_dir
model_name = 'model_99_0711.json'
scores = []
for train, test in skf.split(x, Y):
    print("Running Fold", c+1, "/", n_folds)
    if log_dir:
        config.tb_log_dir = log_dir + '/fold' + str(c)
    score = main.train(config, model_name,
                       x[train], y[train], x[test], y[test])
    print('Acc: %.2f%% \nLoss: %.2f' % (score[1]*100, score[0]))
    scores.append(score[1]*100)

    logging.info('*'*30)
    logging.info(score)
    logging.info('*'*30)
    c += 1
mean = np.mean(scores)
std = np.std(scores)
logging.info('Mean %f' % mean)
logging.info('Std %f' % std)
print('Acc: %.2f%% \t Std: %.2f' % (mean, std))
