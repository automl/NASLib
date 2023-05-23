import logging
import os
import json
from naslib.evaluators.zc_ensemble_evaluator import ZCEnsembleEvaluator
from naslib.search_spaces import get_search_space
from naslib.utils.get_dataset_api import get_dataset_api
from naslib.utils.log import setup_logger
from naslib import utils
from naslib.search_spaces.nasbench201.conversions import convert_str_to_op_indices as convert_nb201_str_to_op_indices

n_archs_to_sample = 50
ALL_ARCHS = True

config = utils.get_config_from_args()

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)

utils.log_args(config)

search_space = get_search_space(config.search_space, config.dataset)
dataset_api = get_dataset_api(config.search_space, config.dataset)

utils.set_seed(config.seed)

archs = set()

if config.search_space == 'nasbench201':
    if ALL_ARCHS == True:
        n_archs_to_sample = len(dataset_api['nb201_data'].keys())
    for idx, arch in enumerate(dataset_api['nb201_data'].keys()):
        if idx >= n_archs_to_sample:
            break
        archs.add(convert_nb201_str_to_op_indices(arch))
elif config.search_space == 'transbench101_micro':
    if ALL_ARCHS == True:
        n_archs_to_sample = len(dataset_api['api'].all_arch_dict['micro'])

    for idx, arch in enumerate(dataset_api['api'].all_arch_dict['micro']):
        if idx >= n_archs_to_sample:
            break
        archs.add(tuple(int(x) for x in arch.split('-')[-1].replace('_', '')))
elif config.search_space == 'transbench101_macro':
    if ALL_ARCHS == True:
        n_archs_to_sample = len(dataset_api['api'].all_arch_dict['macro'])

    for idx, arch in enumerate(dataset_api['api'].all_arch_dict['macro']):
        if idx >= n_archs_to_sample:
            break
        archs.add(tuple(int(x) for x in arch.split('-')[1].replace('_', '')))
else:
    # ZC_TODO: Handle other search spaces (301, 101)
    while len(archs) < n_archs_to_sample:
        encodings = []
        for i in range(5):
            graph = search_space.clone()
            graph.sample_random_architecture(dataset_api)
            encodings.append(str(graph.get_hash()))

        archs.update(encodings)
        logger.info(f'Sampled {len(archs)} unique models so far')

archs_dict = {idx: str(arch) for idx, arch in enumerate(archs)}

save_file = f'archs_{config.search_space}.json'
with open(os.path.join(config.data, save_file), 'w') as f:
    json.dump(archs_dict, f)
logger.info('Done.')
