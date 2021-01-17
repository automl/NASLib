# Author: Robin Ru @ University of Oxford
# This is an implementation of learning curve extrapolation method based on:
# B. Baker et al. 2017, “Accelerating neural architecture search using performance prediction,” arXiv preprint arXiv:1705.10823.

from sklearn.svm import NuSVR
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
import time
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np
from naslib.predictors.predictor import Predictor
from scipy import stats
import copy
import numpy as np
from naslib.predictors.lcsvr import loguniform

from naslib.search_spaces.core.query_metrics import Metric

class Aug_SVR_Estimator(Predictor):

    def __init__(self, metric=Metric.VAL_ACCURACY, all_curve=True, zero_cost_methods=['jacov'], config=None,
                 model_name='svr',best_hyper=None, n_hypers=1000):
        # metric is a list for Aug SVR e.g.[Metric.VAL_ACCURACY, ]

        self.n_hypers = n_hypers
        self.all_curve = all_curve
        self.model_name = model_name
        self.best_hyper = best_hyper
        self.metric = metric

        self.config = config
        self.zero_cost_methods = zero_cost_methods
        if len(self.zero_cost_methods) > 0:
            self.include_zero_cost = True
        else:
            self.include_zero_cost = False

    def pre_process(self):
        if self.include_zero_cost:
            # pre load image training data for zero-cost methods
            from naslib.utils import utils
            self.train_loader, _, _, _, _ = utils.get_train_val_loaders(self.config, mode='train')

    def pre_compute(self, xtrain, xtest):

        if self.include_zero_cost:
            # compute zero-cost scores for test and train data
            from naslib.predictors.zerocost_estimators import ZeroCostEstimators
            self.xtrain_zc_scores = {}
            self.xtest_zc_scores = {}
            for method_name in self.zero_cost_methods:
                print(f'pre-compute {method_name} scores for all train and test data')
                zc_method = ZeroCostEstimators(self.config, batch_size=64, method_type=method_name)
                zc_method.train_loader = copy.deepcopy(self.train_loader)
                xtrain_zc_scores = zc_method.query(xtrain)
                xtest_zc_scores = zc_method.query(xtest)

                self.xtrain_zc_scores[f'{method_name}_scores'] = list(xtrain_zc_scores)
                self.xtest_zc_scores[f'{method_name}_scores'] = list(xtest_zc_scores)

        else:
            self.xtrain_zc_scores = None
            self.xtest_zc_scores = None

    def fit(self, xtrain, ytrain, info, learn_hyper=True):

        # prepare training data
        xtrain_data = self.prepare_data(info, zero_cost_info=self.xtrain_zc_scores)
        y_train = np.array(ytrain)

        # learn hyperparameters of the extrapolator by cross validation
        if self.best_hyper is None or learn_hyper:
            # specify model hyper-parameters
            if self.model_name == 'svr':
                C = loguniform(1e-5, 10, self.n_hypers)
                nu = np.random.uniform(0, 1, self.n_hypers)
                gamma = loguniform(1e-5, 10, self.n_hypers)
                hyper = np.vstack([C, nu, gamma]).T
            elif self.model_name == 'blr':
                alpha_1 = np.random.uniform(1e-7, 1e-5, self.n_hypers)
                alpha_2 = np.random.uniform(1e-7, 1e-5, self.n_hypers)
                lambda_1 = np.random.uniform(1e-7, 1e-5, self.n_hypers)
                lambda_2 = np.random.uniform(1e-7, 1e-5, self.n_hypers)
                hyper = np.vstack([alpha_1, alpha_2, lambda_1, lambda_2]).T
            elif self.model_name == 'rf':
                n_trees = np.random.randint(10, 800, self.n_hypers)
                frac_feature = np.random.uniform(0.1, 0.5, self.n_hypers)
                hyper = np.vstack([n_trees, frac_feature]).T

            print(f'start CV on {self.model_name}')
            mean_score_list = []
            t_start = time.time()
            for i in range(self.n_hypers):
                # define model
                if self.model_name == 'svr':
                    model = NuSVR(C=hyper[i, 0], nu=hyper[i, 1], gamma=hyper[i, 2], kernel='rbf')
                    # model = SVR(C=hyper[i, 0], nu=hyper[i, 1], gamma= ,kernel='linear')
                elif self.model_name == 'blr':
                    model = BayesianRidge(alpha_1=hyper[i, 0], alpha_2=hyper[i, 1],
                                          lambda_1=hyper[i, 2], lambda_2=hyper[i, 3])
                elif self.model_name == 'rf':
                    model = RandomForestRegressor(n_estimators=int(hyper[i, 0]), max_features=hyper[i, 1])
                # perform cross validation to learn the best hyper value
                scores = cross_val_score(model, xtrain_data, y_train, cv=3)
                mean_scores = np.mean(scores)
                mean_score_list.append(mean_scores)
                # print(f'hper={hyper[i]}, score={mean_scores}')
            t_end = time.time()
            best_hyper_idx = np.argmax(mean_score_list)
            best_hyper = hyper[best_hyper_idx]
            max_score = np.max(mean_score_list)
            time_taken = t_end - t_start
            print(f'{self.model_name}'
                  f'best_hyper={best_hyper}, score={max_score}, time={time_taken}')
            self.best_hyper = best_hyper

        # fit the extrapolator with the best hyperparameters to the training data
        if self.model_name == 'svr':
            best_model = NuSVR(C=self.best_hyper[0], nu=self.best_hyper[1], gamma=self.best_hyper[2], kernel='rbf')
            # model = SVR(C=hyper[i, 0], nu=hyper[i, 1], gamma= ,kernel='linear')
        elif self.model_name == 'blr':
            best_model = BayesianRidge(alpha_1=self.best_hyper[0], alpha_2=self.best_hyper[1],
                                       lambda_1=self.best_hyper[2], lambda_2=self.best_hyper[3])
        elif self.model_name == 'rf':
            best_model = RandomForestRegressor(n_estimators=int(self.best_hyper[0]), max_features=self.best_hyper[1])

        best_model.fit(xtrain_data, y_train)
        self.best_model = best_model

    def query(self, xtest, info):
        data = self.prepare_data(info, zero_cost_info=self.xtest_zc_scores)
        pred_on_test_set = self.best_model.predict(data)
        return pred_on_test_set
    
    def get_data_reqs(self):
        """
        Returns a dictionary with info about whether the predictor needs
        extra info to train/query.
        """
        reqs = {'requires_partial_lc':True, 
                'metric':self.metric, 
                'requires_hyperparameters':True, 
                'hyperparams':['flops', 'latency', 'params']
               }
        return reqs

    def prepare_data(self, info, zero_cost_info=None):
        # todo: this can be added at the top of collate_inputs
        arch_params = []
        all_metrics_info = {f'{lc_metric.name}_lc': [] for lc_metric in self.metric}

        for i in range(len(info)):
            for lc_metric in all_metrics_info.keys():
                lc_metric_score = info[i][lc_metric]
                all_metrics_info[lc_metric].append(lc_metric_score)
            arch_hp = [info[i][hp] for hp in ['flops', 'latency', 'params']]
            arch_params.append(arch_hp)

        all_metrics_info['arch_params'] = arch_params
        if self.include_zero_cost:
            for zc_method in self.zero_cost_methods:
                all_metrics_info[f'{zc_method}_scores'] = zero_cost_info[f'{zc_method}_scores']

        return self.collate_inputs(all_metrics_info)

    def collate_inputs(self, all_metrics_info):

        TS_list = []
        for metric_name, metric_content in all_metrics_info.items():
            if 'lc' in metric_name:
                # learning curve metrics like valid acc and train losses
                lc_metric_list = metric_content
                lc_metric = np.vstack(lc_metric_list)  # dimension: n_archs x n_epochs
                Dlc_metric = np.diff(lc_metric, n=1, axis=1)
                DDlc_metric = np.diff(Dlc_metric, n=1, axis=1)

                mlc_metric = np.mean(lc_metric, axis=1)[:, None]
                stdlc_metric = np.std(lc_metric, axis=1)[:, None]
                mDlc_metric = np.mean(Dlc_metric, axis=1)[:, None]
                stdDlc_metric = np.std(Dlc_metric, axis=1)[:, None]
                mDDlc_metric = np.mean(DDlc_metric, axis=1)[:, None]
                stdDDlc_metric = np.std(DDlc_metric, axis=1)[:, None]

                if self.all_curve:
                    collated_metric_list = [lc_metric, Dlc_metric, DDlc_metric, mlc_metric, stdlc_metric]
                else:
                    collated_metric_list = [mlc_metric, stdlc_metric, mDlc_metric, stdDlc_metric, mDDlc_metric, stdDDlc_metric]

                if 'TRAIN_LOSS' in metric_name:
                    sumVC = np.sum(lc_metric, axis=1)[:, None]
                    collated_metric_list += [sumVC]

            else:
                # architecture parameters like flops and latency or zc scores
                arch_metric_list = metric_content
                if len(arch_metric_list) != 0:
                    AP = np.vstack(arch_metric_list)
                    collated_metric_list = [AP]
                else:
                    collated_metric_list = []

            TS_list += collated_metric_list

        X = np.hstack(TS_list)

        return X
