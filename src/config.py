







# Here are some basic settings.
# It could be overwritten if you want to specify
# some configs. However, please check the correspoding
# codes in loadopts.py.



import torch
import logging
from .dict2obj import Config



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = "../data" # the path saving the data
DOWNLOAD = False # whether to download the data

SAVED_FILENAME = "paras.pt" # the filename of saved model paramters
PRE_BEST = "best"
INFO_PATH = "./infos/{method}/{dataset}-{model}/{description}"
LOG_PATH = "./logs/{method}/{dataset}-{model}/{description}-{time}"
TIMEFMT = "%m%d%H"

# logger
LOGGER = Config(
    name='DFK', filename='log.txt', level=logging.DEBUG,
    filelevel=logging.DEBUG, consolelevel=logging.INFO,
    formatter=Config(
        filehandler=logging.Formatter('%(asctime)s:\t%(message)s'),
        consolehandler=logging.Formatter('%(message)s')
    )
)


# default transforms
TRANSFORMS = {
    'voc2007': 'voc,none',
    'voc2012': 'voc,none',
}


# env settings
NUM_WORKERS = 3
PIN_MEMORY = True

# basic properties of inputs
MEANS = {
    "mnist": [0.,],
    "fashionmnist": [0.,],
    'svhn': [0.5, 0.5, 0.5],
    "cifar10": [0.4914, 0.4824, 0.4467],
    "cifar100": [0.5071, 0.4867, 0.4408],
    "tinyimagenet": [0.4601, 0.4330, 0.3732],
    "voc2012": [0.5, 0.5, 0.5] # TODO
}

STDS = {
    "mnist": [1.,],
    "fashionmnist": [1.,],
    'svhn': [0.5, 0.5, 0.5],
    "cifar10": [0.2471, 0.2435, 0.2617],
    "cifar100": [0.2675, 0.2565, 0.2761],
    "tinyimagenet": [0.2647, 0.2481, 0.2594],
    "voc2012": [0.5, 0.5, 0.5] # TODO
}

# the settings of optimizers of which lr could be pointed
# additionally.
OPTIMS = {
    "sgd": Config(lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=False, prefix="SGD:"),
    "adam": Config(lr=0.01, betas=(0.9, 0.999), weight_decay=0., prefix="Adam:")
}


# the learning schedule can be added here
LEARNING_POLICY = {
    "default": (
        "StepLR",
        Config(
            step_size=3,
            gamma=0.33,
            prefix="default leaning policy will be applied:"
        )
    ),
    "null": (
        "StepLR",
        Config(
            step_size=9999999999999,
            gamma=1,
            prefix="Null leaning policy will be applied:"
        )
    ),
   "template": (
        "MultiStepLR",
        Config(
            milestones=[100, 105],
            gamma=0.1,
            prefix="Template leaning policy will be applied:"
        )
    ),
    "cosine":(   
        "CosineAnnealingLR",   
        Config(          
            T_max=200,
            eta_min=0.,
            last_epoch=-1,
            prefix="cosine learning policy: T_max == epochs - 1:"
        )
    )
}






