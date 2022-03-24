import logging

from naslib.evaluators.zc_evaluator import PredictorEvaluator
from naslib.predictors import ZeroCost
from naslib.search_spaces import get_search_space
from naslib.utils import utils, setup_logger, get_dataset_api


config = utils.get_config_from_args()
utils.set_seed(config.seed)
logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)
utils.log_args(config)

#If the API did not evaluate *all* architectures in the search space,
#set load_labeled=True
config_search_spaces = ["transbench101_micro", "darts", "nlp"] ##
predictors = ["fisher", "flops", "grad_norm", "grasp", "jacov", "params", "snip", "synflow"] ##
dataset_dic = {
    'transbench101': ['jigsaw', 'class_scene', 'class_object', 'autoencoder'],
    'nlp': 
}
for config_search_space in config_search_spaces:

    config.search_space = config_search_space
    load_labeled = True if config.search_space in ["darts", "nlp"] else False

    dataset_api = get_dataset_api(config.search_space, config.dataset)

    # initialize the search space and predictor
    utils.set_seed(config.seed)
    # can be "fisher", "grasp", "grad_norm", "jacov" "snip", "synflow", "flops",
    # "params"
    for p in predictors:
        config.predictor = p
        predictor = ZeroCost(config, batch_size=config.batch_size,
                     method_type=config.predictor)
        search_space = get_search_space(name=config.search_space, dataset=config.dataset)

        # initialize the PredictorEvaluator class
        predictor_evaluator = PredictorEvaluator(predictor, config=config)
        predictor_evaluator.adapt_search_space(
            search_space, load_labeled=load_labeled, dataset_api=dataset_api
        )

        # evaluate the predictor
        predictor_evaluator.evaluate()
