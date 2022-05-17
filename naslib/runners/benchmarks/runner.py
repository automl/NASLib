import logging
import timeit
import os
import json

from naslib.predictors.zerocost import ZeroCost
from naslib.search_spaces import get_search_space
from naslib.search_spaces.core.query_metrics import Metric
from naslib.utils.get_dataset_api import get_dataset_api, load_sampled_architectures
from naslib.utils.logging import setup_logger
from naslib.utils import utils

config = utils.get_config_from_args()

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)

utils.log_args(config)

search_space = get_search_space(config.search_space, config.dataset)
dataset_api = get_dataset_api(config.search_space, config.dataset)
archs = load_sampled_architectures(config.search_space, config.dataset)
archs_to_evaluate = {idx: eval(archs[str(idx)]) for idx in range(config.start_idx, config.start_idx + config.n_models)}

utils.set_seed(config.seed)
train_loader, _, _, _, _ = utils.get_train_val_loaders(config)

predictor = ZeroCost(method_type=config.predictor)

for idx, arch in archs_to_evaluate.items():
    logger.info(f'Computing ZC score for model {idx} with encoding {arch}')
    zc_score = {}
    graph = search_space.clone()
    graph.set_spec(arch)
    graph.parse()
    accuracy = graph.query(Metric.VAL_ACCURACY, config.dataset, dataset_api=dataset_api)

    # Query predictor
    start_time = timeit.default_timer()
    score = predictor.query(graph, train_loader)
    end_time = timeit.default_timer()

    zc_score['idx'] = str(idx)
    zc_score['arch'] = str(arch)
    zc_score[predictor.method_type] = {
        'score': score,
        'time': end_time - start_time
    }
    zc_score['val_accuracy'] = accuracy

    output_dir = os.path.join(config.data, 'zc_benchmarks', config.search_space, config.dataset)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, f'benchmark_{idx}.json')

    with open(output_file, 'w') as f:
        json.dump(zc_score, f)

logger.info('Done.')
