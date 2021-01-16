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
from naslib.predictors.lcsvr import SVR_Estimator

from naslib.search_spaces.core.query_metrics import Metric

class Aug_SVR_Estimator(SVR_Estimator):

    def __init__(self, metric=Metric.VAL_ACCURACY, all_curve=True, zero_cost_to_include=[], model_name='svr',best_hyper=None, n_hypers=1000):
        # metric is a list for Aug SVR e.g.[Metric.VAL_ACCURACY, ]
        super().__init__(metric)

        self.n_hypers = n_hypers
        self.all_curve = all_curve
        self.model_name = model_name
        self.best_hyper = best_hyper
        self.name = 'LcSVR'
        self.metric = metric

    def pre_process(self):
        print('get jacobian scores for all train and test data')

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
                # architecture parameters like flops and latency
                arch_metric_list = metric_content
                if len(arch_metric_list) != 0:
                    AP = np.vstack(arch_metric_list)
                    collated_metric_list = [AP]
                else:
                    collated_metric_list = []

            TS_list += collated_metric_list

        X = np.hstack(TS_list)

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
                'metric':self.metric, 
                'requires_hyperparameters':True, 
                'hyperparams':['flops', 'latency', 'params']
               }
        return reqs
        
    def prepare_data(self, info):
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

        return self.collate_inputs(all_metrics_info)
