import main
import utils
import numpy as np
from sklearn.model_selection import StratifiedKFold

class Config(main.Config):
    # Data settings
    dataset_dir = '../dataset_padded'
    labels_filename = 'Y_truth.txt'
    input_shape = (256, 256, 1)
    out_dim = (2,)

    # Network settings
    epochs = 1  #

    # GA settings
    model_type = 'cnn'
    early=False
    
    tb_log_dir = '/content/IA/My Drive/IA/carie_class/logs/test_del'
    
config = Config()

x, Y, y,  = utils.load_dataset(config, split=False)
n_folds = 10
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1)

c = 0
log_dir = config.tb_log_dir
model_name = 'model.json'
scores = []
for train, test in skf.split(x, Y):
    print( "Running Fold", c+1, "/", n_folds )
    c+=1
    config.tb_log_dir = log_dir + '/fold' + str(c+1)
    score = main.train(config, model_name,
                        x[train], y[train], x[test], y[test])
    print('Acc: %.f2%% \nLoss: %.2f' % (score[1]*100, score[0]) )
    scores.append(score[1]*100)
mean = np.mean( scores )
std = np.std( scores )

print( 'Acc: %.2f%% \t Std: %.2f' % (mean, std) )
    
    
