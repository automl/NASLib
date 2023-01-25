import logging
import timeit
import os
import json

from naslib.predictors.zerocost import ZeroCost
from naslib.search_spaces import get_search_space
from naslib.search_spaces.core.query_metrics import Metric
from naslib.utils.get_dataset_api import get_dataset_api, load_sampled_architectures
from naslib.utils.log import setup_logger
from naslib import utils

def translate_str(s, replace_str='[]', with_str='()'):
    table = str.maketrans(replace_str, with_str)
    return str.translate(s, table)

config = utils.get_config_from_args()
logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)

utils.log_args(config)

search_space = get_search_space(config.search_space, config.dataset)
dataset_api = get_dataset_api(config.search_space, config.dataset)

if config.dataset in ['ninapro', 'svhn', 'scifar100']:
    postfix = '9x'
    with open(f'./naslib/data/9x/{config.search_space}/{config.dataset}/test.json') as f:
        api9x_data = json.load(f)
    api9x = {translate_str(str(record['arch'])): record['accuracy'] for record in api9x_data}
else:
    postfix = ''

archs = load_sampled_architectures(config.search_space, postfix)
end_index = config.start_idx + config.n_models if config.start_idx + config.n_models < len(archs) else len(archs)
archs_to_evaluate = {idx: eval(archs[str(idx)]) for idx in range(config.start_idx, end_index)}

utils.set_seed(config.seed)
train_loader, _, _, _, _ = utils.get_train_val_loaders(config)

predictor = ZeroCost(method_type=config.predictor)

zc_scores = []

for i, (idx, arch) in enumerate(archs_to_evaluate.items()):
    try:
        logger.info(f'{i} \tComputing ZC score for model id {idx} with encoding {arch}')
        zc_score = {}
        graph = search_space.clone()
        graph.set_spec(arch)
        graph.parse()
        if config.dataset in ['ninapro', 'svhn', 'scifar100']:
            accuracy = api9x[str(arch)]
        else:
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
        zc_scores.append(zc_score)

        output_dir = os.path.join(config.data, 'zc_benchmarks', config.predictor)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_file = os.path.join(output_dir, f'benchmark--{config.search_space}--{config.dataset}--{config.start_idx}.json')

        with open(output_file, 'w') as f:
            json.dump(zc_scores, f)
    except Exception as e:
        logger.info(f'Failed to compute score for model {idx} with arch {arch}!')

logger.info('Done.')
