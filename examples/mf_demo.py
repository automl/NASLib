# purpose of this script:
# debug random search optimizer to better understand function

from naslib import search_spaces
from naslib.optimizers.discrete.sh import optimizer
from naslib.search_spaces import NasBench201SearchSpace as NB201

import logging
from naslib.utils import utils, setup_logger, get_dataset_api

from naslib.optimizers import RandomSearch as RS
from naslib.optimizers import RegularizedEvolution as RE
from naslib.optimizers import SuccessiveHalving as SH
from naslib.optimizers import HyperBand as HB

from naslib.defaults.trainer_multifidelity import Trainer
#from naslib.defaults.trainer import Trainer

import yaml
from pathlib import Path
import os

import mf_plot

from shutil import copyfile

demo_config = None
with open(os.path.join(str(Path(__file__).parent), 'mf_demo.yaml'), "r") as stream:
    try:
        demo_config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

config_optimizer = demo_config['optimizer']
config_path_optimizer = os.path.join(
    utils.get_project_root(), "benchmarks", "nas_predictors", "discrete_config"
) + "_" + config_optimizer + ".yaml"
default_config_path = os.path.join(
    utils.get_project_root(), "benchmarks", "nas_predictors", "discrete_config.yaml"
) 

copyfile(config_path_optimizer, default_config_path)

# init search space
search_space = NB201()

# read config
config = utils.get_config_from_args(config_type="nas_predictor")
utils.set_seed(config.seed)
utils.log_args(config)

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)

# define optimizer

if config_optimizer == 'SH':
    optimizer = SH(config)
    plot_func = mf_plot.plot_sh
elif config_optimizer == 'HB':
    optimizer = HB(config)
    plot_func = mf_plot.plot_hb
elif config_optimizer == 'RS':
    optimizer = RS(config)
elif config_optimizer == 'BOHB':
    optimizer = HB(config)
else:
    raise Exception('invalid config')

# load nasbench data, there data seems to be generalised
dataset_api = get_dataset_api(config.search_space, config.dataset)

# adapt search space
optimizer.adapt_search_space(search_space, dataset_api=dataset_api)

## Running search with Trainer
trainer = Trainer(optimizer, config, lightweight_output=True)

# run search for number of iterations specified
trainer.search()

trainer.evaluate(dataset_api=dataset_api)

if demo_config['plot'] == False:
    exit()

plot_func()