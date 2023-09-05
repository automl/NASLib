import logging

from naslib.predictors import ZeroCost
from naslib.search_spaces import NasBench101SearchSpace, NasBench201SearchSpace, NasBench301SearchSpace
from naslib.optimizers import  Bananas, Npenas
from naslib.defaults.trainer import Trainer
from naslib.search_spaces.core import Metric
from naslib import utils
from naslib.utils.dataset import get_train_val_loaders
from naslib.utils import get_config_from_args, setup_logger, get_dataset_api
from naslib.utils import compute_scores, get_zc_benchmark_api


config = get_config_from_args(config_type='zc')

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)

utils.log_args(config)

supported_search_spaces = {
    'nasbench101': NasBench101SearchSpace,
    'nasbench201': NasBench201SearchSpace,
    'nasbench301': NasBench301SearchSpace,
}
supported_optimizers = {
    "bananas": Bananas,
    "npenas": Npenas,
}

dataset_api = get_dataset_api(config.search_space, config.dataset)
utils.set_seed(config.seed)

# Getting started with Zero cost Predictors

# The Zero Cost predictors can be any of the following:
# {'epe_nas', 'fisher', 'grad_norm', 'grasp', 'jacov', 'l2_norm', 'nwot', 'plain', 
#      'snip', 'synflow', 'zen', 'flops', 'params'}

graph = supported_search_spaces[config.search_space]()
train_loader, val_loader, test_loader, train_transform, valid_transform = get_train_val_loaders(config)

zc_pred = config.predictor # "fisher" # "plain"
graph.sample_random_architecture(dataset_api=dataset_api)
graph.parse()
zc_predictor = ZeroCost(method_type=zc_pred)
zc_score = zc_predictor.query(graph=graph, dataloader=train_loader)

print(f'Score of model for Zero Cost predictor {zc_pred}: {zc_score}')

# Check correlation score
# Metrics available: ["mae", "rmse", "pearson", "spearman", "kendalltau", "kt_2dec", 
#                       "kt_1dec", "full_ytest", "full_testpred"]

val_accs = []
zc_scores = []
for _ in range(10):
    graph = supported_search_spaces[config.search_space]()
    graph.sample_random_architecture()
    graph.parse()
    acc = graph.query(metric=Metric.VAL_ACCURACY, dataset='cifar10',
                      dataset_api=dataset_api)
    val_accs.append(acc)
    zc_score = zc_predictor.query(graph=graph, dataloader=train_loader)
    zc_scores.append(zc_score)

corr_score = compute_scores(ytest=val_accs, test_pred=zc_scores)

print("The kendall-tau score is: ", corr_score["kendalltau"])


# Calculating zero-cost scores can also take a while. Use zc api
zc_api = get_zc_benchmark_api(config.search_space, config.dataset)
graph = supported_search_spaces[config.search_space]()
graph.sample_random_architecture()

# Use the Zero Cost Benchmark to get the score for the model for a particular ZC proxy
zc_predictor = 'jacov'
spec = graph.get_hash()
zc_score = zc_api[str(spec)][zc_predictor]['score']
time_to_compute = zc_api[str(spec)][zc_predictor]['time']

print(f'All the data available in the Zero Cost benchmark for model {spec}: ')
print(zc_api[str(spec)][zc_predictor])
print(f'Score of model with spec {spec} for Zero Cost proxies {zc_predictor}: {zc_score}')
print(f'Time taken to compute the score for the model: {time_to_compute:.2f}s')

# Using inside training 
# Only bananas and npenas has zero cost implemented inside their query funcitons
search_space = supported_search_spaces[config.search_space]()

# optimizer = NPENAS(config, zc_api=zc_api)
optimizer = supported_optimizers[config.optimizer](config, zc_api=zc_api) 
optimizer.adapt_search_space(search_space, dataset_api=dataset_api)

trainer = Trainer(optimizer, config, lightweight_output=True)

trainer.search(resume_from="")
trainer.evaluate(resume_from="", dataset_api=dataset_api)
