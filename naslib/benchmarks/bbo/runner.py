import logging
import sys
#from nasbench import api

from naslib.defaults.trainer import Trainer
from naslib.optimizers import RandomSearch, Npenas, \
RegularizedEvolution, LocalSearch, Bananas, BasePredictor

from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces import NasBench101SearchSpace, NasBench201SearchSpace, \
DartsSearchSpace, NasBenchNLPSearchSpace, TransBench101SearchSpace
from naslib.utils import utils, setup_logger, get_dataset_api

from torch.utils.tensorboard import SummaryWriter

config = utils.get_config_from_args(config_type='nas')

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)

utils.log_args(config)

writer = SummaryWriter(config.save)

supported_optimizers = {
    'rs': RandomSearch(config),
    're': RegularizedEvolution(config),
    'bananas': Bananas(config),
    'npenas': Npenas(config),
    'ls': LocalSearch(config),
}

supported_search_spaces = {
    'nasbench101': NasBench101SearchSpace(),
    'nasbench201': NasBench201SearchSpace(),
    'darts': DartsSearchSpace(),
    'nlp': NasBenchNLPSearchSpace(),
    'transbench101': TransBench101SearchSpace()
    # 'nasbenchasr': NasBenchASRSearchSpace()
}

dataset_api = get_dataset_api(config.search_space, config.dataset)
utils.set_seed(config.seed)

search_space = supported_search_spaces[config.search_space]

metric = Metric.VAL_ACCURACY if config.search_space == 'darts' else None

optimizer = supported_optimizers[config.optimizer]
optimizer.adapt_search_space(search_space, dataset_api=dataset_api)

trainer = Trainer(optimizer, config, lightweight_output=True)

trainer.search(resume_from="", summary_writer=writer, report_incumbent=False)
trainer.evaluate(resume_from="", dataset_api=dataset_api, metric=metric)