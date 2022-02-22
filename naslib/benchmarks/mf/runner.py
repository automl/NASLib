import logging
import sys
#from nasbench import api

from naslib.defaults.trainer import Trainer
from naslib.defaults.trainer_multifidelity import Trainer as Trainer_MF

from naslib.optimizers import RandomSearch, Npenas, \
RegularizedEvolution, LocalSearch, Bananas, BasePredictor, SuccessiveHalving, HB

from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces import NasBench101SearchSpace, NasBench201SearchSpace, \
DartsSearchSpace, NasBenchNLPSearchSpace, TransBench101SearchSpace, NasBenchASRSearchSpace
from naslib.utils import utils, setup_logger, get_dataset_api

from torch.utils.tensorboard import SummaryWriter

config = utils.get_config_from_args(config_type='bbo-bs')

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)

utils.log_args(config)

# writer = SummaryWriter(config.save)

supported_optimizers = {
    'rs': RandomSearch(config),
    're': RegularizedEvolution(config),
    'bananas': Bananas(config),
    'npenas': Npenas(config),
    'ls': LocalSearch(config),
    'sh': SuccessiveHalving(config),
    # 'hb': HyperBand(config),
}

supported_search_spaces = {
    'nasbench101': NasBench101SearchSpace(),
    'nasbench201': NasBench201SearchSpace(),
    'darts': DartsSearchSpace(),
    'nlp': NasBenchNLPSearchSpace(),
    'transbench101_micro': TransBench101SearchSpace(),
    'transbench101_macro': TransBench101SearchSpace(),
    'asr': NasBenchASRSearchSpace(),
}

dataset_api = get_dataset_api(config.search_space, config.dataset)
utils.set_seed(config.seed)

search_space = supported_search_spaces[config.search_space]
if config.search_space == 'transbench101_macro':
    search_space.space = 'macro'

metric = Metric.VAL_ACCURACY if config.search_space == 'darts' else None

optimizer = supported_optimizers[config.optimizer]
optimizer.adapt_search_space(search_space, dataset_api=dataset_api)

trainer = Trainer(optimizer, config, lightweight_output=True)
multi_fidelity_optimizers = {'sh', 'hb'}
if config.optimizer in multi_fidelity_optimizers:
    trainer = Trainer_MF(optimizer, config, lightweight_output=True)
# trainer.search(resume_from="", summary_writer=writer, report_incumbent=False)
trainer.search(resume_from="", report_incumbent=False)
trainer.evaluate(resume_from="", dataset_api=dataset_api, metric=metric)

# error: FileNotFoundError: [Errno 2] No such file or directory: '/Users/lars/Projects/NASLib/naslib/data/nasbench_only108.pkl'
