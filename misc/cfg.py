from yacs.config import CfgNode as CN



CFG = CN()

# Configuration Settings

# dataset info
CFG.dataset = 'cifar10'
CFG.data_dir = '<data path>'
CFG.size = 224 
CFG.nclass = 10
CFG.nch = 3

# if dataset is ImageNet subsets
CFG.dseed = 0 # random seed to sample classes
CFG.rrc = True

# Condensing configs --------------------------------------------------------
    # model
CFG.net_type = 'convnet'
CFG.depth = 3
CFG.width = 1.0
CFG.niter = 5000
CFG.ipm = 5 # iteration per (random) model
CFG.batch_real = 128
CFG.batch_syn_max = 200
CFG.lr_img = 5e-3
CFG.mom_img = 0.5
CFG.weight_decay = 5e-4


# Synthetic Set configs -----------------------------------------------------
CFG.ipc = 1
CFG.factor = 2
CFG.decode_type = 'single'
CFG.init = 'random'
CFG.aug_type = 'color_crop_cutout'
CFG.load_memory = True # load training images to the memory
CFG.norm_type = 'instance' # choose from 'batch', 'instance', 'sn' and 'none'

# Training configs -----------------------------------------------------------
CFG.epochs = 1000
CFG.batch_size = 64
CFG.workers = 8
CFG.lr = 1e-2
CFG.momentum = 0.9
CFG.seed = 42
    # Mixup
CFG.mixup = 'cut' # choose from 'cut' and "vanilla"
CFG.beta = 1.0 # mixup beta distribution
CFG.mix_p = 1.0 # mixup probability
    # dsa
CFG.dsa = True
CFG.dsa_strategy = 'color_crop_cutout_flip_scale_rotate'
CFG.augment = False
    # Kernel type of the RKHS
CFG.kernel = 'gaussian'

# Logging configs -------------------------------------------------------------
CFG.epoch_eval_interval = 500 # interval when evaluating the synthetic data
CFG.results_path = '<path to results>'
CFG.epoch_print_freq = 100
CFG.test_it_interval = 500


