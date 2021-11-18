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

# init search space
search_space = NB201()

# read config
config = utils.get_config_from_args(config_type="nas_predictor")
utils.set_seed(config.seed)
utils.log_args(config)

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)

# define optimizer 
#optimizer = SH(config)
#optimizer = RS(config)
optimizer = HB(config)
# load nasbench data, there data seems to be generalised
dataset_api = get_dataset_api(config.search_space, config.dataset)

# adapt search space
optimizer.adapt_search_space(search_space, dataset_api=dataset_api)


## Running search with Trainer

trainer = Trainer(optimizer, config, lightweight_output=True)

# run search for number of iterations specified
trainer.search()

trainer.evaluate(dataset_api=dataset_api)