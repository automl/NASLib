import unittest
import numpy as np
import torch
from naslib.defaults.predictor_evaluator import PredictorEvaluator
from naslib.search_spaces import NasBench201SearchSpace
from naslib.predictors import BayesianLinearRegression, BOHAMIANN, GPPredictor, RandomForestPredictor, LGBoost, \
    XGBoost, NGBoost
from naslib import utils
from naslib.utils import get_dataset_api

x_data = np.load('assets/nb201_test_set_x.npy', allow_pickle=True)
y_data = np.load('assets/nb201_test_set_y.npy', allow_pickle=True)
info_data = np.load('assets/nb201_test_set_info.npy', allow_pickle=True)
times_data = np.load('assets/nb201_test_set_times.npy', allow_pickle=True)
test_idx = int(np.ceil(len(x_data) * 0.7))
train_data = (x_data[:test_idx], y_data[:test_idx], info_data[:test_idx], times_data[:test_idx])
test_data = (x_data[test_idx:], y_data[test_idx:], info_data[test_idx:], times_data[test_idx:])
supported_search_spaces = {
    'nasbench201': NasBench201SearchSpace()
}


class PredictorsTest(unittest.TestCase):

    def setUp(self):
        self.args = utils.parse_args(args=['--config-file', 'assets/test_predictor.yaml'])
        self.config = utils.get_config_from_args(self.args, config_type='predictor')
        self.search_space = supported_search_spaces[self.config.search_space]
        utils.set_seed(self.config.seed)
        self.load_labeled = (True if self.config.search_space in ['nasbench301', 'nlp'] else False)
        self.dataset_api = get_dataset_api(search_space='test')

    def test_configFile(self):
        self.assertEqual(self.config.search_space, 'nasbench201')
        self.assertEqual(self.config.dataset, 'cifar10')
        self.assertEqual(self.config.test_size, 10)
        self.assertEqual(self.config.train_size_single, 20)
        self.assertEqual(self.config.seed, 1000)
        self.assertEqual(self.config.uniform_random, 1)

    def test_RFPredictor(self):
        predictor = RandomForestPredictor(encoding_type=None)
        predictor_evaluator = PredictorEvaluator(predictor, config=self.config)
        predictor_evaluator.adapt_search_space(self.search_space, load_labeled=self.load_labeled,
                                               dataset_api=self.dataset_api)
        predictor_evaluator.single_evaluate(train_data=train_data, test_data=test_data,
                                            fidelity=self.config.fidelity_single)

        self.assertAlmostEqual(predictor_evaluator.results[-1]['kendalltau'], 0.4666666666666, places=3)
        self.assertAlmostEqual(predictor_evaluator.results[-1]['mae'], 4.7984154693486, places=3)
        self.assertAlmostEqual(predictor_evaluator.results[-1]['rmse'], 5.400776348052, places=3)
        self.assertAlmostEqual(predictor_evaluator.results[-1]['pearson'], 0.7369233227443, places=3)
        self.assertAlmostEqual(predictor_evaluator.results[-1]['spearman'], 0.5428571428571, places=3)

    def test_LGBPredictor(self):
        predictor = LGBoost(encoding_type=None)
        predictor_evaluator = PredictorEvaluator(predictor, config=self.config)
        predictor_evaluator.adapt_search_space(self.search_space, load_labeled=self.load_labeled,
                                               dataset_api=self.dataset_api)
        predictor_evaluator.single_evaluate(train_data=train_data, test_data=test_data,
                                            fidelity=self.config.fidelity_single)
        self.assertAlmostEqual(predictor_evaluator.results[-1]['kendalltau'], 0.11547005383792, places=3)
        self.assertAlmostEqual(predictor_evaluator.results[-1]['mae'], 7.726079796113, places=3)
        self.assertAlmostEqual(predictor_evaluator.results[-1]['rmse'], 7.905554796885, places=3)
        self.assertAlmostEqual(predictor_evaluator.results[-1]['pearson'], 0.06237005477731, places=3)
        self.assertAlmostEqual(predictor_evaluator.results[-1]['spearman'], 0.13093073414159, places=3)

    def test_BLRPredictor(self):
        predictor = BayesianLinearRegression(encoding_type=None)
        predictor_evaluator = PredictorEvaluator(predictor, config=self.config)
        predictor_evaluator.adapt_search_space(self.search_space, load_labeled=self.load_labeled,
                                               dataset_api=self.dataset_api)
        predictor_evaluator.single_evaluate(train_data=train_data, test_data=test_data,
                                            fidelity=self.config.fidelity_single)
        self.assertAlmostEqual(predictor_evaluator.results[-1]['kendalltau'], -0.7333333333333, places=3)
        self.assertAlmostEqual(predictor_evaluator.results[-1]['mae'], 14.308725383199, places=3)
        self.assertAlmostEqual(predictor_evaluator.results[-1]['rmse'], 16.09739638867, places=3)
        self.assertAlmostEqual(predictor_evaluator.results[-1]['pearson'], 0.6415900987808, places=3)
        self.assertAlmostEqual(predictor_evaluator.results[-1]['spearman'], -0.8857142857142, places=3)

    def test_BohamiannPredictor(self):
        predictor = BOHAMIANN(encoding_type=None)
        predictor_evaluator = PredictorEvaluator(predictor, config=self.config)
        predictor_evaluator.adapt_search_space(self.search_space, load_labeled=self.load_labeled,
                                               dataset_api=self.dataset_api)
        predictor_evaluator.single_evaluate(train_data=train_data, test_data=test_data,
                                            fidelity=self.config.fidelity_single)
        self.assertAlmostEqual(predictor_evaluator.results[-1]['kendalltau'], 0.3333333333333, places=3)
        self.assertAlmostEqual(predictor_evaluator.results[-1]['mae'], 4.948659613539, places=3)
        self.assertAlmostEqual(predictor_evaluator.results[-1]['rmse'], 6.423134627655, places=3)
        self.assertAlmostEqual(predictor_evaluator.results[-1]['pearson'], 0.39805559977643, places=3)
        self.assertAlmostEqual(predictor_evaluator.results[-1]['spearman'], 0.6, places=3)

    def test_XGBPredictor(self):
        predictor = XGBoost(encoding_type=None)
        predictor_evaluator = PredictorEvaluator(predictor, config=self.config)
        predictor_evaluator.adapt_search_space(self.search_space, load_labeled=self.load_labeled,
                                               dataset_api=self.dataset_api)
        predictor_evaluator.single_evaluate(train_data=train_data, test_data=test_data,
                                            fidelity=self.config.fidelity_single)
        self.assertAlmostEqual(predictor_evaluator.results[-1]['kendalltau'], 0.6, places=3)
        self.assertAlmostEqual(predictor_evaluator.results[-1]['mae'], 3.68074010213, places=3)
        self.assertAlmostEqual(predictor_evaluator.results[-1]['rmse'], 4.7317729522208, places=3)
        self.assertAlmostEqual(predictor_evaluator.results[-1]['pearson'], 0.6612804210411, places=3)
        self.assertAlmostEqual(predictor_evaluator.results[-1]['spearman'], 0.7142857142857, places=3)

    def test_NGBPredictor(self):
        predictor = NGBoost(encoding_type=None)
        predictor_evaluator = PredictorEvaluator(predictor, config=self.config)
        predictor_evaluator.adapt_search_space(self.search_space, load_labeled=self.load_labeled,
                                               dataset_api=self.dataset_api)
        predictor_evaluator.single_evaluate(train_data=train_data, test_data=test_data,
                                            fidelity=self.config.fidelity_single)  
        self.assertAlmostEqual(predictor_evaluator.results[-1]['kendalltau'], 0.2, places=3)
        self.assertAlmostEqual(predictor_evaluator.results[-1]['mae'], 5.72692900534, places=3)
        self.assertAlmostEqual(predictor_evaluator.results[-1]['rmse'], 6.7191690329, places=3)
        self.assertAlmostEqual(predictor_evaluator.results[-1]['pearson'], 0.333039889, places=3)
        self.assertAlmostEqual(predictor_evaluator.results[-1]['spearman'], 0.37142857142, places=3)
        
    def test_GPPredictor(self):
        predictor = GPPredictor(encoding_type=None, optimize_gp_hyper=False)
        predictor_evaluator = PredictorEvaluator(predictor, config=self.config)
        predictor_evaluator.adapt_search_space(self.search_space, load_labeled=self.load_labeled,
                                               dataset_api=self.dataset_api)
        predictor_evaluator.single_evaluate(train_data=train_data, test_data=test_data,
                                            fidelity=self.config.fidelity_single)
        self.assertAlmostEqual(predictor_evaluator.results[-1]['kendalltau'], 0.7453559924999, places=3)
        self.assertAlmostEqual(predictor_evaluator.results[-1]['mae'], 4.41141527697, places=3)
        self.assertAlmostEqual(predictor_evaluator.results[-1]['rmse'], 5.990029403098, places=3)
        self.assertAlmostEqual(predictor_evaluator.results[-1]['pearson'], 0.9519836508794, places=3)
        self.assertAlmostEqual(predictor_evaluator.results[-1]['spearman'], 0.8196885999705, places=3)


if __name__ == '__main__':
    unittest.main()
