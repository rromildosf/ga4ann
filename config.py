class Config():
    # Data settings
    dataset_dir = 'PATH_TO_DATASET'
    labels_filename = 'labels.txt' # Name of labels of dataset
    # subset = {'train': 'train_masks'} # Use it if you dataset Y are masks
    input_shape = (256, 256, 1) # tuple of image shape or tuple of output size. i.e. (32,32,3) or (10,)
    out_dim = (2,) # tuple of mask shape or tuple of output size. i.e. (10,)

    # Network settings
    epochs = 1000  # not exactly if use early=True
    batch_size = 10 # only used if use_generator=True
    steps_per_epoch = 100 # only used if use_generator=True
    validation_steps = 10 # only used if use_generator=True
    use_generator = False

    # GA settings
    model_type = 'cnn' # 'ann' or 'cnn'
    generations = 30
    population = 30

    #general settings
    verbose = 1
    min_acc = 0.7 #only save a model file if acc of model is great than or equal to this value
    early = True # early stopping
    tb_log_dir = '/content/IA/My Drive/carie_seg/tb_logs/aug4/' # directory to log tensorboard files, if None don't save
