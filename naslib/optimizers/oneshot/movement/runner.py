import logging
from pyexpat import model
import sys
import naslib as nl


from naslib.defaults.trainer import Trainer
from naslib.optimizers import (
    DARTSOptimizer,
    GDASOptimizer,
    OneShotNASOptimizer,
    RandomNASOptimizer,
    RandomSearch,
    RegularizedEvolution,
    LocalSearch,
    Bananas,
    BasePredictor,
    GSparseOptimizer,
    GMovementOptimizer,
    MovementOptimizer,
    DrNASOptimizer,
)

from naslib.search_spaces import NasBench201SearchSpace, DartsSearchSpace, NasBench101SearchSpace
from naslib.utils import utils, setup_logger, get_dataset_api
from naslib.search_spaces.core.query_metrics import Metric

config = utils.get_config_from_args()
utils.set_seed(config.seed)

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)  # default DEBUG is too verbose

utils.log_args(config)

supported_optimizers = {
    "darts": DARTSOptimizer(config),
    "gdas": GDASOptimizer(config),
    "oneshot": OneShotNASOptimizer(config),
    "rsws": RandomNASOptimizer(config),
    "re": RegularizedEvolution(config),
    "rs": RandomSearch(config),
    "ls": RandomSearch(config),
    "bananas": Bananas(config),
    "bp": BasePredictor(config),
    "gsparsity": GSparseOptimizer(config),
    "gmovement": GMovementOptimizer(config),
    "movement": MovementOptimizer(config),
    "drnas": DrNASOptimizer(config)
}

if config.dataset =='cifar100':
    num_classes=100
elif config.dataset=='ImageNet16-120':
    num_classes=120
else:
    num_classes=10
supported_search_space ={
    "nasbench201" : NasBench201SearchSpace(num_classes),
    "darts" : DartsSearchSpace(num_classes),
    "nasbench101" : NasBench101SearchSpace(num_classes)
}

#search_space = NasBench201SearchSpace()
search_space = supported_search_space[config.search_space]
#dataset_api = get_dataset_api("nasbench201", config.dataset)
#print(search_space)
#dataset_api = get_dataset_api(config.search_space, config.dataset)
dataset_api = get_dataset_api(config.search_space, config.dataset)

optimizer = supported_optimizers[config.optimizer]
optimizer.adapt_search_space(search_space)

trainer = Trainer(optimizer, config, lightweight_output=True)
trainer.search()

# if not config.eval_only:
#    checkpoint = utils.get_last_checkpoint(config) if config.resume else ""
#    trainer.search(resume_from=checkpoint)

#checkpoint = utils.get_last_checkpoint(config, search_model=True) if config.resume else ""
#trainer.evaluate(resume_from=checkpoint, dataset_api=dataset_api)
mov_model="/work/dlclarge2/agnihotr-ml/NASLib/naslib/optimizers/oneshot/movement/run/nasbench201/cifar10/movement/14/search/model_final.pth"
darts_model="/work/dlclarge2/agnihotr-ml/NASLib/naslib/optimizers/oneshot/movement/run/nasbench201/cifar10/darts/10/search/model_final.pth"
gdas_model="/work/dlclarge2/agnihotr-ml/NASLib/naslib/optimizers/oneshot/movement/run/nasbench201/cifar10/gdas/10/search/model_final.pth"
drnas_model="/work/dlclarge2/agnihotr-ml/NASLib/naslib/optimizers/oneshot/gmovement/run/nasbench201/cifar10/drnas/10/search/model_final.pth"
model = drnas_model
#model = "/work/dlclarge2/agnihotr-ml/NASLib/naslib/optimizers/oneshot/movement/run/darts/cifar10/darts/10/search/model_final.pth"
#trainer.evaluate(dataset_api=dataset_api, metric=Metric.VAL_ACCURACY, search_model=model)
trainer.evaluate(dataset_api=dataset_api, metric=Metric.VAL_ACCURACY)#, search_model=model)
