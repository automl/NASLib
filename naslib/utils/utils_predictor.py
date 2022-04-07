
from naslib.evaluators.zc_evaluator import ZeroCostPredictorEvaluator
from naslib.predictors import ZeroCost
from naslib.search_spaces import get_search_space
from naslib.utils import utils, setup_logger, get_dataset_api
import argparse
from fvcore.common.config import CfgNode

def evaluate_predictor_across_search_spaces(config):
        kendalltau_avg=0
        count=0
        search_spaces=["nasbench201", "nasbench301", "transbench101_micro"]
        datasets={"nasbench201":["cifar10","cifar100","ImageNet16-120"],"nasbench301":["cifar10"],"transbench101_micro":["jigsaw", "class_object","class_scene"]}
        kendalltau_dict={}
        for ss in search_spaces:
          for ds in datasets[ss]:
            config.dataset=ds
            config.search_space=ss
            dataset_api = get_dataset_api(ss,ds)
            config.train_data_file = None
            config.test_data_file = None
            config.save = "{}/{}/{}/{}/{}".format(config.out_dir,config.dataset,"predictors",config.predictor,config.seed,)
            predictor = ZeroCost(method_type=config.predictor)
            search_space = get_search_space(name=ss, dataset=ds)
            predictor_evaluator = ZeroCostPredictorEvaluator(predictor, config=config)
            predictor_evaluator.adapt_search_space(search_space, dataset_api=dataset_api)
            # Evaluate the predictor
            predictor_evaluator.evaluate()
            kt=predictor_evaluator.results[1]["kendalltau"]
            kendalltau_dict[(config.search_space,config.dataset)]=kt
            kendalltau_avg+=kt
            count=count+1
        for x in kendalltau_dict.keys():
            print("{:25s}||{:25s}||{}".format(x[0],x[1],kendalltau_dict[x]))
        print("KendallTau across all",kendalltau_avg/count)
