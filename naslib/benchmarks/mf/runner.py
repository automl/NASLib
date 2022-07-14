import logging

#from nasbench import api

from naslib.defaults.trainer import Trainer
from naslib.defaults.trainer_multifidelity import Trainer as Trainer_MF

from naslib.optimizers import RandomSearch, Npenas, \
RegularizedEvolution, LocalSearch, Bananas, SuccessiveHalving, HB, BOHB, DEHB

from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces import NasBench201SearchSpace, NasBenchASRSearchSpace
from naslib.utils import utils, setup_logger, get_dataset_api

# from torch.utils.tensorboard import SummaryWriter

config = utils.get_config_from_args(config_type='bbo-bs')

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)

utils.log_args(config)

# writer = SummaryWriter(config.save)

supported_optimizers = {
    'rs': RandomSearch,
    're': RegularizedEvolution,
    'bananas': Bananas,
    'npenas': Npenas,
    'ls': LocalSearch,
    'sh': SuccessiveHalving,
    'hb': HB,
    'bohb': BOHB,
    'dehb': DEHB,
}

supported_search_spaces = {
    'nasbench201': NasBench201SearchSpace,    
    'asr': NasBenchASRSearchSpace,
}

dataset_api = get_dataset_api(config.search_space, config.dataset)
utils.set_seed(config.seed)

search_space = supported_search_spaces[config.search_space]()

metric = Metric.VAL_ACCURACY if config.search_space == 'darts' else None

optimizer = supported_optimizers[config.optimizer](config)
optimizer.adapt_search_space(search_space, dataset_api=dataset_api)

trainer = Trainer(optimizer, config, lightweight_output=True)
multi_fidelity_optimizers = {'sh', 'hb', 'bohb', 'dehb'}
if config.optimizer in multi_fidelity_optimizers:
    trainer = Trainer_MF(optimizer, config, lightweight_output=True)
# trainer.search(resume_from="", summary_writer=writer, report_incumbent=False)
trainer.search(resume_from="")
trainer.evaluate(resume_from="", dataset_api=dataset_api)
