import logging
import sys
import os
import naslib as nl


from naslib.defaults.predictor_evaluator import PredictorEvaluator

from naslib.predictors import Ensemble, FeedforwardPredictor, GBDTPredictor, \
EarlyStopping, GCNPredictor, BonasPredictor, ZeroCostEstimators, SoLosspredictor, \
SVR_Estimator, XGBoost, NGBoost, RandomForestPredictor, DNGOPredictor, \
BOHAMIANN, BayesianLinearRegression, LCNetPredictor, SemiNASPredictor, \
GPPredictor, SparseGPPredictor, VarSparseGPPredictor, \
LCEPredictor, OmniPredictor, OmniXGBPredictor

from naslib.search_spaces import NasBench101SearchSpace, NasBench201SearchSpace, DartsSearchSpace
from naslib.search_spaces.core.query_metrics import Metric

from naslib.utils import utils, setup_logger, get_dataset_api
from naslib.utils.utils import get_project_root


config = utils.get_config_from_args(config_type='predictor')
utils.set_seed(config.seed)
logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)

utils.log_args(config)

supported_predictors = {
    'bananas': Ensemble(predictor_type='bananas', num_ensemble=3),
    'feedforward': FeedforwardPredictor(encoding_type='adjacency_one_hot'),
    'gbdt': GBDTPredictor(encoding_type='adjacency_one_hot'),
    'gcn': GCNPredictor(encoding_type='gcn'),
    'bonas': BonasPredictor(encoding_type='bonas'),
    'valloss': EarlyStopping(metric=Metric.VAL_LOSS),
    'valacc': EarlyStopping(metric=Metric.VAL_ACCURACY),
    'lce': LCEPredictor(metric=Metric.VAL_ACCURACY),
    'jacov': ZeroCostEstimators(config, batch_size=64, method_type='jacov'),
    'snip': ZeroCostEstimators(config, batch_size=64, method_type='snip'),
    'grad_norm': ZeroCostEstimators(config, batch_size=64, method_type='grad_norm'),
    'fisher': ZeroCostEstimators(config, batch_size=64, method_type='fisher'),
    'grasp': ZeroCostEstimators(config, batch_size=64, method_type='grasp'),
    'synflow': ZeroCostEstimators(config, batch_size=64, method_type='synflow'),
    'sotl': SoLosspredictor(metric=Metric.TRAIN_LOSS, sum_option='SoTL'),
    'sotle': SoLosspredictor(metric=Metric.TRAIN_LOSS, sum_option='SoTLE'),
    'sotlema': SoLosspredictor(metric=Metric.TRAIN_LOSS, sum_option='SoTLEMA'),
    'xgb': XGBoost(encoding_type='adjacency_one_hot'),
    'ngb': NGBoost(encoding_type='adjacency_one_hot'),
    'rf': RandomForestPredictor(encoding_type='adjacency_one_hot'),
    'dngo': DNGOPredictor(encoding_type='adjacency_one_hot'),
    'bohamiann': BOHAMIANN(encoding_type='adjacency_one_hot'),
    'lcnet': LCNetPredictor(encoding_type='adjacency_one_hot'),
    'bayes_lin_reg': BayesianLinearRegression(encoding_type='adjacency_one_hot'),
    'seminas': SemiNASPredictor(encoding_type='seminas', semi=True),
    'nao': SemiNASPredictor(encoding_type='seminas', semi=False),
    'gp': GPPredictor(encoding_type='adjacency_one_hot'),
    'sparse_gp': SparseGPPredictor(encoding_type='adjacency_one_hot',
                                   optimize_gp_hyper=True, num_steps=100),
    'var_sparse_gp': VarSparseGPPredictor(encoding_type='adjacency_one_hot',
                                          optimize_gp_hyper=True, num_steps=200),
    'lcsvr': SVR_Estimator(metric=Metric.VAL_ACCURACY, all_curve=False),
    'gbdt_path': GBDTPredictor(encoding_type='path'),
    'ngb_path': NGBoost(encoding_type='path'),
    'dngo_path': DNGOPredictor(encoding_type='path'),
    'bohamiann_path': BOHAMIANN(encoding_type='path'),
    'bayes_lin_reg_path': BayesianLinearRegression(encoding_type='path'),
    'gp_path': GPPredictor(encoding_type='path'),
    'omni': OmniPredictor(zero_cost=['jacov'], lce=['sotle', 'valacc'], encoding_type='adjacency_one_hot', 
                          config=config),
    'omni_xgb': OmniXGBPredictor(zero_cost=['jacov'], lce=[], encoding_type='adjacency_one_hot', 
                                 config=config),
    'omni_both': OmniPredictor(zero_cost=['jacov', 'snip'], lce=['sotle', 'valacc'], encoding_type='adjacency_one_hot',
                               config=config),
    'omni_lofi': OmniPredictor(zero_cost=['jacov'], lce=[], encoding_type='adjacency_one_hot', 
                               config=config, run_pre_compute=True, min_train_size=0),
    'omni_no_zero': OmniPredictor(zero_cost=[], lce=['sotle'], encoding_type='adjacency_one_hot',
                                  config=config, run_pre_compute=False, min_train_size=0),
    'omni_no_enc': OmniPredictor(zero_cost=['jacov'], lce=['sotle'], encoding_type=None,
                                 config=config, run_pre_compute=True, min_train_size=0),
    'ngb_hp': OmniPredictor(zero_cost=[], lce=[], encoding_type='adjacency_one_hot',
                            config=config, run_pre_compute=False, min_train_size=0),
    'xgb_hpo': XGBoost(encoding_type='adjacency_one_hot', hpo_wrapper=True),
    'feedforward_hpo': FeedforwardPredictor(encoding_type='adjacency_one_hot', hpo_wrapper=True),
}

supported_search_spaces = {
    'nasbench101': NasBench101SearchSpace(),
    'nasbench201': NasBench201SearchSpace(),
    'darts': DartsSearchSpace()
}

load_labeled = (True if config.search_space == 'darts' else False)
dataset_api = get_dataset_api(config.search_space, config.dataset)
utils.set_seed(config.seed)

# set up the search space and predictor
predictor = supported_predictors[config.predictor]
search_space = supported_search_spaces[config.search_space]
predictor_evaluator = PredictorEvaluator(predictor, config=config)
predictor_evaluator.adapt_search_space(search_space, load_labeled=load_labeled, dataset_api=dataset_api)

# evaluate the predictor
predictor_evaluator.evaluate()
