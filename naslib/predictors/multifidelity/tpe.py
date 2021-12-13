import statsmodels.api as sm
from collections import defaultdict
import numpy as np
from collections import defaultdict




class TreeParserEstimator(): #TODO maybe this need 
    def __init__(self):
        self.p = 0.15
        self.N_min = 0
        self.min_points_in_model = 0
        self.configs = defaultdict(lambda: defaultdict(list))
        self.losses = defaultdict(lambda: defaultdict(list))
        self.top_n_percent = 15
        




    def train(self,model):
        budget = self.fidelity
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

        n_good= max(self.min_points_in_model, (self.top_n_percent * train_configs.shape[0])//100 )
        n_bad = max(self.min_points_in_model, ((100-self.top_n_percent)*train_configs.shape[0])//100)

        train_data_good = self.impute_conditional_data(train_configs[idx[:n_good]])
        train_data_bad  = self.impute_conditional_data(train_configs[idx[n_good:n_good+n_bad]])
        
        # quick rule of thumb
        bw_estimation = 'normal_reference'
        bad_kde = sm.nonparametric.KDEMultivariate(data=train_data_bad,  var_type="u", bw=bw_estimation) #var_type is type of hyperparameter
        # c : continuous, u : unordered (discrete),  o : ordered (discrete)
        good_kde = sm.nonparametric.KDEMultivariate(data=train_data_good, var_type="u", bw=bw_estimation) 
        bad_kde.bw = np.clip(bad_kde.bw, self.min_bandwidth,None)
        good_kde.bw = np.clip(good_kde.bw, self.min_bandwidth,None)

        self.kde_models[budget] = {
                'good': good_kde,
                'bad' : bad_kde
        }

    def impute_conditional_data(self, array):
        return_array = np.empty_like(array)
        for i in range(array.shape[0]):
            datum = np.copy(array[i])
            nan_indices = np.argwhere(np.isnan(datum)).flatten()

            while (np.any(nan_indices)):
                nan_idx = nan_indices[0]
                valid_indices = np.argwhere(np.isfinite(array[:,nan_idx])).flatten()
                if len(valid_indices) > 0:  
                # pick one of them at random and overwrite all NaN values
                    row_idx = np.random.choice(valid_indices)
                    datum[nan_indices] = array[row_idx, nan_indices]

                else:
                    # no good point in the data has this value activated, so fill it with a valid but random value
                    t = self.vartypes[nan_idx]
                    if t == 0:
                        datum[nan_idx] = np.random.rand()
                    else:
                        datum[nan_idx] = np.random.randint(t)
                nan_indices = np.argwhere(np.isnan(datum)).flatten()
            return_array[i,:] = datum
        return(return_array)
   