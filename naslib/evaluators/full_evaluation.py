import os

from .zc_evaluator import ZeroCostPredictorEvaluator
from naslib.predictors import Predictor
from naslib.search_spaces import get_search_space
from naslib.utils import get_dataset_api, get_config_from_args

DATASETS = {
    "nasbench201": [
        "cifar10",
        "cifar100",
        "ImageNet16-120"],
    "nasbench301": ["cifar10"],
    "transbench101_micro": [
        "jigsaw",
        "class_object",
        "class_scene"]
	}

def full_evaluate_predictor(predictor:Predictor, test_size:int=100, search_spaces=("nasbench201", "nasbench301", "transbench101_micro")) -> None:
    """ Evaluate a predictor for all the supported tasks of a given search space

    Args:
        predictor       : Zero cost predictor to evaluate
        test_size       : Number of models to sample and rank using the predictor per search-space/task combination
        search_spaces   : Search space to search

    Return:
        None
    """
    # Load the default configs
    config = get_config_from_args(args=None)

    scores = {}

    for search_space_name in search_spaces:
        for dataset in DATASETS[search_space_name]:
            # Update config for this search space and dataset
            config.dataset = dataset
            config.search_space = search_space_name
            config.test_size = test_size
            config.save = os.path.join(config.out_dir, config.dataset, "predictors", config.predictor, str(config.seed))

            # Get benchmark API and search space graph
            dataset_api = get_dataset_api(search_space_name, dataset)
            search_space = get_search_space(name=search_space_name, dataset=dataset)

            # Initialize and adapt the ZeroCostPredictorEvaluator to the search space
            predictor_evaluator = ZeroCostPredictorEvaluator(predictor, config=config)
            predictor_evaluator.adapt_search_space(search_space, dataset_api=dataset_api)

            # Evaluate the predictor
            predictor_evaluator.evaluate()
            kt = predictor_evaluator.results[1]["kendalltau"]

            scores[(search_space_name, dataset)] = kt

    for x in scores.keys():
        print("{:25s}||{:25s}||{}".format(x[0], x[1], scores[x]))

    kt_values = scores.values()
    print("Average Kendall-Tau:", sum(kt_values) / len(kt_values))
