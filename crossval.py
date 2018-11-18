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
    epochs = 30  #

    # GA settings
    model_type = 'cnn'
    early=False
    
    tb_log_dir = '/content/IA/My Drive/IA/carie_class/logs/'
    
config = Config()

x, Y, y,  = train.load_dataset(config, split=False)
n_folds = 10
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1)
c = 0
for train, test in skf.split(x, Y):
    print( "Running Fold", c+1, "/", n_folds )
    c+=1
    config.tb_log_dir = config.tb_log_dir + '/fold' + str(c+1)
    score = main.train(config, 'model.json',
                                x[train], y[train], x[test], y[test])

    
    
