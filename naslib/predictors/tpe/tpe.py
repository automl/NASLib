import statsmodels.api as sm
from collections import defaultdict
import numpy as np

from naslib.predictors.utils.encodings import encode
import scipy.stats as sps
import scipy.optimize as spo

import torch #maybe not sure rigth now

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
        self.search_space = None #thing about later
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
    
    def query(self, budget):
        """
            Function to sample a new configuration
            This function is called inside Hyperband to query a new configuration
            Parameters:
            -----------
            budget: float
                the budget for which this configuration is scheduled
            returns: config
                should return a valid configuration
        """
        sample = None
        info_dict = {}
        
        # If no model is available, sample from prior
        # also mix in a fraction of random configs
        if len(self.kde_models.keys()) == 0 or np.random.rand() < self.random_fraction:
            sample =  self.configspace.sample_configuration()
            info_dict['model_based_pick'] = False
        sample =  torch.nn.Module()  # hacky way to get arch and accuracy checkpointable
        sample.arch = self.search_space.clone()
        best = np.inf
        best_vector = None

        if sample is None:
            try:
                
                #sample from largest budget
                budget = max(self.kde_models.keys())

                l = self.kde_models[budget]['good'].pdf
                g = self.kde_models[budget]['bad' ].pdf
            
                minimize_me = lambda x: max(1e-32, g(x))/max(l(x),1e-32)
                
                kde_good = self.kde_models[budget]['good']
                kde_bad = self.kde_models[budget]['bad']

                for i in range(self.num_samples):
                    idx = np.random.randint(0, len(kde_good.data))
                    datum = kde_good.data[idx]
                    vector = []
                    
                    for m,bw,t in zip(datum, kde_good.bw, self.vartypes):
                        
                        bw = max(bw, self.min_bandwidth)
                        if t == 0:
                            bw = self.bw_factor*bw
                            try:
                                vector.append(sps.truncnorm.rvs(-m/bw,(1-m)/bw, loc=m, scale=bw))
                            except:
                                self.logger.warning("Truncated Normal failed for:\ndatum=%s\nbandwidth=%s\nfor entry with value %s"%(datum, kde_good.bw, m))
                                self.logger.warning("data in the KDE:\n%s"%kde_good.data)
                        else:
                            
                            if np.random.rand() < (1-bw):
                                vector.append(int(m))
                            else:
                                vector.append(np.random.randint(t))
                    val = minimize_me(vector)

                    if not np.isfinite(val):
                        self.logger.warning('sampled vector: %s has EI value %s'%(vector, val))
                        self.logger.warning("data in the KDEs:\n%s\n%s"%(kde_good.data, kde_bad.data))
                        self.logger.warning("bandwidth of the KDEs:\n%s\n%s"%(kde_good.bw, kde_bad.bw))
                        self.logger.warning("l(x) = %s"%(l(vector)))
                        self.logger.warning("g(x) = %s"%(g(vector)))

                        # right now, this happens because a KDE does not contain all values for a categorical parameter
                        # this cannot be fixed with the statsmodels KDE, so for now, we are just going to evaluate this one
                        # if the good_kde has a finite value, i.e. there is no config with that value in the bad kde, so it shouldn't be terrible.
                        if np.isfinite(l(vector)):
                            best_vector = vector
                            break

                    if val < best:
                        best = val
                        best_vector = vector

                if best_vector is None:
                    self.logger.debug("Sampling based optimization with %i samples failed -> using random configuration"%self.num_samples)
                    sample.arch.sample_random_architecture(dataset_api=self.dataset_api) 
                    info_dict['model_based_pick']  = False
                else:
                    self.logger.debug('best_vector: {}, {}, {}, {}'.format(best_vector, best, l(best_vector), g(best_vector)))
                    #for i, hp_value in enumerate(best_vector):
                    #    if isinstance(
                    #       self.configspace.get_hyperparameter(
                    #            self.configspace.get_hyperparameter_by_idx(i)
                    #       ),
                    #        ConfigSpace.hyperparameters.CategoricalHyperparameter
                    #   ):
                    #       best_vector[i] = int(np.rint(best_vector[i]))
                    sample.arch.convert_op_indices_to_naslib(best_vector)
            except:
                self.logger.warning("Sampling based optimization with %i samples failed\n %s \nUsing random configuration"%(self.num_samples, traceback.format_exc()))
                sample = self.configspace.sample_configuration()
                info_dict['model_based_pick']  = False


  
        self.logger.debug('done sampling a new configuration.')
        return sample, info_dict
    
    
    

 