""" Evaluates a ZeroCost predictor for a search space and dataset/task"""
import logging
import os

from naslib.evaluators.zc_evaluator import ZeroCostPredictorEvaluator
from naslib.predictors import ZeroCost
from naslib.search_spaces import get_search_space
from naslib.utils import utils, setup_logger, get_dataset_api
import argparse
from fvcore.common.config import CfgNode
parser = argparse.ArgumentParser()
parser.add_argument('--predictor', default='synflow', type=str, help='predictor')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--test_size', default=5, type=int, help='test_size')
parser.add_argument('--train_size', default=5, type=int, help='train_size')
parser.add_argument('--out_dir', default='run/', type=str, help='out dir')
parser.add_argument('--seed', default=1000, type=int, help='seed')
parser.add_argument('--cutout', default=False, type=bool, help='cutout')
parser.add_argument('--cutout_length', default=16, type=int, help='cutout')
parser.add_argument('--cutout_prob', default=1.0, type=float, help='cutout')
parser.add_argument('--train_portion', default=0.7, type=float, help='train_portion')
config = parser.parse_args()
kendalltau_all=0
count=0
config=CfgNode(vars(config))
config.data = os.path.join(utils.get_project_root(), 'data')

# Set the search spaces and datasets to avergage over
search_spaces=["nasbench201", "nasbench301", "transbench101_micro"]
datasets={"nasbench201":["cifar10","cifar100","ImageNet16-120"],"nasbench301":["cifar10"],"transbench101_micro":["jigsaw", "class_object","class_scene"]}
for ss in search_spaces:
    for ds in datasets[ss]:
        config.dataset=ds
        config.search_space=ss
        dataset_api = get_dataset_api(ss,ds)
        config.train_data_file = None
        config.test_data_file = None
        config.save = "{}/{}/{}/{}/{}".format(config.out_dir,config.dataset,"predictors",config.predictor,config.seed,)
        logger = setup_logger(config.save + "/log.log")
        logger.setLevel(logging.INFO)
        # Initialize the search space and predictor
        # Method type can be "fisher", "grasp", "grad_norm", "jacov", "snip", "synflow", "flops" or "params"
        predictor = ZeroCost(method_type=config.predictor)
        search_space = get_search_space(name=ss, dataset=ds)
  
        # Initialize the ZeroCostPredictorEvaluator class
        predictor_evaluator = ZeroCostPredictorEvaluator(predictor, config=config)
        predictor_evaluator.adapt_search_space(search_space, dataset_api=dataset_api)
        # Evaluate the predictor
        predictor_evaluator.evaluate()
        kendalltau_all+=predictor_evaluator.results[1]["kendalltau"]
        count=count+1
print("KendallTau across all",kendalltau_all/count)

