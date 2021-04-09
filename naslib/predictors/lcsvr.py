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
import numpy as np

from naslib.search_spaces.core.query_metrics import Metric

def loguniform(low=0, high=1, size=None):
    return np.exp(np.random.uniform(np.log(low), np.log(high), size))


class SVR_Estimator(Predictor):

    def __init__(self, metric=Metric.VAL_ACCURACY, all_curve=True, model_name='svr',best_hyper=None, n_hypers=1000, require_hyper=True):

        self.n_hypers = n_hypers
        self.all_curve = all_curve
        self.model_name = model_name
        self.best_hyper = best_hyper
        self.name = 'LcSVR'
        self.metric=metric
        self.require_hyperparameters = require_hyper

    def fit(self, xtrain, ytrain, info, learn_hyper=True):

        # prepare training data
        xtrain_data = self.prepare_data(info)
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

    def collate_inputs(self, VC_all_archs_list, AP_all_archs_list):
        """
        Args:
            VC_all_archs_list: a list of validation accuracy curves for all archs
            AP_all_archs_list: a list of architecture features for all archs

        Returns:
            X: an collated array of all input information used for extrapolation model

        """
        VC = np.vstack(VC_all_archs_list)  # dimension: n_archs x n_epochs
        DVC = np.diff(VC, n=1, axis=1)
        DDVC = np.diff(DVC, n=1, axis=1)

        mVC = np.mean(VC, axis=1)[:, None]
        stdVC = np.std(VC, axis=1)[:, None]
        mDVC = np.mean(DVC, axis=1)[:, None]
        stdDVC = np.std(DVC, axis=1)[:, None]
        mDDVC = np.mean(DDVC, axis=1)[:, None]
        stdDDVC = np.std(DDVC, axis=1)[:, None]

        if self.all_curve:
            TS_list = [VC, DVC, DDVC, mVC, stdVC]
        else:
            TS_list = [mVC, stdVC, mDVC, stdDVC, mDDVC, stdDDVC]

        if self.metric == Metric.TRAIN_LOSS:
            sumVC = np.sum(VC, axis=1)[:, None]
            TS_list += [sumVC]

        TS = np.hstack(TS_list)

        if len(AP_all_archs_list) != 0:
            AP = np.vstack(AP_all_archs_list)
            X = np.hstack([AP, TS])
        else:
            X = TS

        return X


    def query(self, xtest, info):
        data = self.prepare_data(info)
        pred_on_test_set = self.best_model.predict(data)        
        return pred_on_test_set
    
    def get_data_reqs(self):
        """
        Returns a dictionary with info about whether the predictor needs
        extra info to train/query.
        """
        reqs = {'requires_partial_lc':True, 
                'metric': self.metric,
                'requires_hyperparameters':self.require_hyperparameters,
                'hyperparams':['flops', 'latency', 'params'],
                'unlabeled':False, 
                'unlabeled_factor':0
               }
        return reqs
        
    def prepare_data(self, info):
        # todo: this can be added at the top of collate_inputs
        val_acc_curve = []
        arch_params = []
        
        for i in range(len(info)):
            acc_metric = info[i]['lc']
            if self.require_hyperparameters:
                arch_hp = [info[i][hp] for hp in ['flops', 'latency', 'params']]
                arch_params.append(arch_hp)
            val_acc_curve.append(acc_metric)
        return self.collate_inputs(val_acc_curve, arch_params)
