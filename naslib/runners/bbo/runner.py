import logging

from naslib.defaults.trainer import Trainer
from naslib.optimizers import Bananas

from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces import get_search_space
from naslib.utils import utils, setup_logger, get_dataset_api

from torch.utils.tensorboard import SummaryWriter

config = utils.get_config_from_args()

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)

utils.log_args(config)

writer = SummaryWriter(config.save)

supported_optimizers = {
    'bananas': Bananas(config)
}

search_space = get_search_space(config.search_space, config.dataset)
dataset_api = get_dataset_api(config.search_space, config.dataset)
utils.set_seed(config.seed)

metric = Metric.VAL_ACCURACY if config.search_space == 'darts' else None

optimizer = supported_optimizers[config.optimizer]
optimizer.adapt_search_space(search_space, dataset_api=dataset_api)

trainer = Trainer(optimizer, config, lightweight_output=True)

trainer.search(resume_from="", summary_writer=writer, report_incumbent=False)

logger('Ensemble experiment complete.')