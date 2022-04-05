import logging
import os
from glob import glob as ls

from naslib.evaluators.zc_evaluator import ZeroCostPredictorEvaluator
from naslib.search_spaces import get_search_space
from naslib.utils import utils, setup_logger
from predictor import ZeroCostPredictor

def get_full_paths_of_files_with_name(root_folder, filename):
    ''' Get the absolute paths of all files with the given filename '''
    return ls(f'{root_folder}/**/{filename}', recursive=True)

config = utils.get_config_from_args()
utils.set_seed(config.seed)
logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)
utils.log_args(config)

data_files = get_full_paths_of_files_with_name('.', 'test.json')

for datafile in data_files:
    config.test_data_file = datafile
    components = config.test_data_file.split(os.sep)
    search_space, dataset = components[2], components[3]

    logger.info(f'Evaluating predictor for {search_space} search space for {dataset} task')
    # initialize the search space and predictor
    predictor = ZeroCostPredictor()
    search_space = get_search_space(name=search_space, dataset=dataset)

    # initialize the ZeroCostPredictorEvaluator class
    predictor_evaluator = ZeroCostPredictorEvaluator(predictor, config=config)
    predictor_evaluator.adapt_search_space(search_space, load_labeled=False)

    # evaluate the predictor
    predictor_evaluator.evaluate()
