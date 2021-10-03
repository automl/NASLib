import logging
import naslib as nl

from naslib.defaults.trainer import Trainer
from naslib.optimizers import (
    DARTSOptimizer,
    GDASOptimizer,
    DrNASOptimizer,
)
from naslib.search_spaces import DartsSearchSpace, NasBench201SearchSpace, NasBench101SearchSpace #, TransBench101SearchSpaceMicro
from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.nasbench101.conversions import convert_naslib_to_spec
from naslib.search_spaces.nasbench201.conversions import convert_naslib_to_str
from naslib.search_spaces.darts.conversions import convert_naslib_to_genotype
#from naslib.search_spaces.transbench101.conversions import convert_naslib_to_transbench101_micro
from naslib.utils import utils, setup_logger, get_dataset_api

from torch.utils.tensorboard import SummaryWriter

config = utils.get_config_from_args()
utils.set_seed(config.seed)

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)  # default DEBUG is too verbose

logger.info(f'Seed          : {config.seed}')
logger.info(f'Search space  : {config.search_space}')
logger.info(f'Optimizer     : {config.optimizer}')
logger.info(f'Save dir      : {config.save}')
utils.log_args(config)

supported_optimizers = {
    "darts": DARTSOptimizer(config),
    "drnas": DrNASOptimizer(config),
    "gdas": GDASOptimizer(config),
}

supported_search_spaces = {
    "nasbench101": NasBench101SearchSpace(),
    "nasbench201": NasBench201SearchSpace(),
    "darts": DartsSearchSpace(),
    #"transbench101_micro": TransBench101SearchSpaceMicro(config.dataset)
}

supported_conversions = {
    "nasbench101": convert_naslib_to_spec,
    "nasbench201": convert_naslib_to_str,
    "darts": convert_naslib_to_genotype,
    #"transbench101_micro": convert_naslib_to_transbench101_micro
}

metrics = {
    "nasbench101": Metric.TEST_ACCURACY,
    "nasbench201": Metric.TEST_ACCURACY,
    "darts": Metric.VAL_ACCURACY,
}

search_space = supported_search_spaces[config.search_space]
dataset_api = get_dataset_api(config.search_space, config.dataset)

optimizer = supported_optimizers[config.optimizer]
optimizer.adapt_search_space(search_space)

writer = SummaryWriter(config.save)

def log_discrete_arch(epoch):
    best_arch = optimizer.get_final_architecture()
    test_accuracy = best_arch.query(metric=metrics[config.search_space], dataset=config.dataset, dataset_api=dataset_api)
    logger.info(f'Best architecture at the end of epoch {epoch}: {supported_conversions[config.search_space](best_arch)}')
    logger.info(f'Test accuracy at the end of epoch {epoch} (queried from benchmark): {test_accuracy}')
    writer.add_scalar('Test accuracy (queried from benchmark)', test_accuracy, epoch)

trainer = Trainer(optimizer, config)
trainer.search(
    resume_from=utils.get_last_checkpoint(config) if config.resume else "",
    summary_writer=writer
    #after_epoch=log_discrete_arch
)

# Log the final architecture and test accuracy
best_arch = optimizer.get_final_architecture()
results = best_arch.query(metric=metrics[config.search_space], dataset=config.dataset, dataset_api=dataset_api)
logger.info(f'Test accuracy of final architecture (queried from benchmark): {results}')
logger.info(f'Best architecture found: {supported_conversions[config.search_space](best_arch)}')