import logging
import sys
import os
import naslib as nl


from naslib.defaults.predictor_evaluator import PredictorEvaluator

from naslib.predictors import Ensemble, FeedforwardPredictor, GBDTPredictor, \
EarlyStopping, GCNPredictor, BonasPredictor, jacobian_cov, SoLosspredictor, \
SVR_Estimator, XGBoost, NGBoost, RandomForestPredictor, DNGOPredictor, \
BOHAMIANN, BayesianLinearRegression, LCNetPredictor, SemiNASPredictor, \
GPPredictor, SparseGPPredictor, VarSparseGPPredictor

from naslib.search_spaces import NasBench101SearchSpace, NasBench201SearchSpace, DartsSearchSpace
from naslib.search_spaces.core.query_metrics import Metric

from naslib.utils import utils, setup_logger, get_dataset_api
from naslib.utils.utils import get_project_root

from fvcore.common.config import CfgNode


config = utils.get_config_from_args(config_type='predictor')

utils.set_seed(config.seed)
logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)

utils.log_args(config)

supported_predictors = {
    'bananas': Ensemble(predictor_type='bananas'),
    'feedforward': FeedforwardPredictor(encoding_type='adjacency_one_hot'),
    'gbdt': GBDTPredictor(encoding_type='adjacency_one_hot'),
    'gcn': GCNPredictor(encoding_type='gcn'),
    'bonas': BonasPredictor(encoding_type='bonas'),
    'valloss': EarlyStopping(metric=Metric.VAL_LOSS),
    'valacc': EarlyStopping(metric=Metric.VAL_ACCURACY),
    'jacov': jacobian_cov(config, batch_size=256),
    'sotl': SoLosspredictor(metric=Metric.TRAIN_LOSS, sum_option='SoTL'),
    'sotle': SoLosspredictor(metric=Metric.TRAIN_LOSS, sum_option='SoTLE'),
    'lcsvr': SVR_Estimator(metric=Metric.VAL_ACCURACY),
    'xgb': XGBoost(encoding_type='adjacency_one_hot'),
    'ngb': NGBoost(encoding_type='adjacency_one_hot'),
    'rf': RandomForestPredictor(encoding_type='adjacency_one_hot'),
    'dngo': DNGOPredictor(encoding_type='adjacency_one_hot'),
    'bohamiann': BOHAMIANN(encoding_type='adjacency_one_hot'),
    'lcnet': LCNetPredictor(encoding_type='adjacency_one_hot'),
    'bayes_lin_reg': BayesianLinearRegression(encoding_type='adjacency_one_hot'),
    'seminas': SemiNASPredictor(encoding_type='seminas'),
    'gp': GPPredictor(encoding_type='adjacency_one_hot'),
    'sparse_gp': SparseGPPredictor(encoding_type='adjacency_one_hot',
                                   optimize_gp_hyper=True, num_steps=100),
    'var_sparse_gp': VarSparseGPPredictor(encoding_type='adjacency_one_hot',
                                          optimize_gp_hyper=True, num_steps=200),
}

supported_search_spaces = {
    'nasbench101': NasBench101SearchSpace(),
    'nasbench201': NasBench201SearchSpace(),
    'darts': DartsSearchSpace()
}

load_labeled = (False if config.search_space == 'nasbench201' else True)
dataset_api = get_dataset_api(config.search_space, config.dataset)

# set up the search space and predictor
predictor = supported_predictors[config.predictor]
search_space = supported_search_spaces[config.search_space]
predictor_evaluator = PredictorEvaluator(predictor, config=config)
predictor_evaluator.adapt_search_space(search_space, load_labeled=load_labeled, dataset_api=dataset_api)

# evaluate the predictor
predictor_evaluator.evaluate()
