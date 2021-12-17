import statsmodels.api as sm
from collections import defaultdict
import numpy as np

from naslib.predictors.utils.encodings import encode




class TreeParserEstimator(): #TODO maybe this need 
    def __init__(
        self,
        encoding_type="adjacency_one_hot",
        ss_type="nasbench201"
    ):
        self.encoding_type = encoding_type
        self.ss_type = ss_type
        self.p = 0.15
        self.N_min = 0
        self.min_points_in_model = 1
        self.configs = defaultdict(lambda: defaultdict(list))
        self.losses = defaultdict(lambda: defaultdict(list))
        self.top_n_percent = 15
        self.min_bandwith = 1e-3
        
    def set_hyperparams(self, _):
        pass



    def fit(self, xtrain, ytrain, train_info):
        
        budget = train_info
        """
        if budget not in self.configs.keys():
            self.configs[budget] = []
            self.losses[budget] = []
        print(model.arch.__str__)
        self.configs[budget].append(model.arch.__str__) #which data has to be in train is to test
        print(model.arch.__str__())
        self.losses[budget].append(model.arch.query(
            self.performance_metric,
            self.dataset,
            epoch=int(self.fidelity),
            dataset_api=self.dataset_api,
        ))
        train_configs = np.array(self.configs[budget])
        train_losses = np.array(self.losses[budget])
        
        # Refit KDE for the current budget
        idx = np.argsort(train_losses)
        """
        if type(xtrain) is list:
            # when used in itself, we use
            xtrain = np.array(
                [
                    [arch.get_op_indices()]
                    for arch in xtrain
                ]
            )
       
        n_good= max(self.min_points_in_model, (self.top_n_percent *len(xtrain)//100)) 
        n_bad = max(self.min_points_in_model, ((100-self.top_n_percent)*len(xtrain)//100))
       
        idx = np.argsort(ytrain)

        train_good_data = xtrain[idx[:n_good]]
        train_bad_data = xtrain[idx[n_good:n_good+n_bad]]
        good_var_type = "".join("u" for i in range(len(train_good_data)))
        bad_var_type = "".join("u" for i in range(len(train_bad_data)))
        # quick rule of thumb
        bw_estimation = 'normal_reference'
        bad_kde = sm.nonparametric.KDEMultivariate(data=train_good_data,  var_type= good_var_type, bw=bw_estimation) #var_type is type of hyperparameter
        # c : continuous, u : unordered (discrete),  o : ordered (discrete)
        good_kde = sm.nonparametric.KDEMultivariate(data=train_bad_data, var_type=  bad_var_type, bw=bw_estimation) 
        bad_kde.bw = np.clip(bad_kde.bw, self.min_bandwidth,None)
        good_kde.bw = np.clip(good_kde.bw, self.min_bandwidth,None)

        self.kde_models[budget] = {
                'good': good_kde,
                'bad' : bad_kde
        }
    

 