import logging

from naslib.defaults.trainer import Trainer
from naslib.optimizers import (
    EdgePopUpOptimizer,
)

from naslib.search_spaces import (
    NasBench101SearchSpace,
)

from naslib.utils import utils, setup_logger, get_dataset_api


config = utils.get_config_from_args(config_type='nas')

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)

utils.log_args(config)

supported_optimizers = {
    'edge_popup': EdgePopUpOptimizer(config),
}

supported_search_spaces = {
    'nasbench101': NasBench101SearchSpace(),
}

dataset_api = get_dataset_api(config.search_space, config.dataset)
utils.set_seed(config.seed)

search_space = supported_search_spaces[config.search_space]

optimizer = supported_optimizers[config.optimizer]
optimizer.adapt_search_space(search_space, dataset_api=dataset_api)


trainer = Trainer(optimizer, config, lightweight_output=True)

trainer.search(resume_from="")
trainer.evaluate(resume_from="", query_benchmark=True, dataset_api=dataset_api)

# run to get acc of the discrete architectures
# for i in range(4, 100, 5):
#     model_path = 'model_'
#     model_path += '0'*6 if i < 10 else '0'*5
#     model_path += str(i) + '.pth'
#     # resume_from = 'run/nasbench301/cifar10/drnas/1/search/model_'
#     # trainer.search(resume_from=model_path)
#     trainer.evaluate(resume_from=model_path, query_benchmark=False, dataset_api=dataset_api)

# run nb101 with validation acc
# trainer.evaluate(resume_from="", query_benchmark=False, metric=Metric.VAL_ACCURACY, dataset_api=dataset_api)
