import numpy as np
import torch
import random
import math

from collections import defaultdict
from naslib.predictors import predictor
from naslib.predictors import ensemble
from naslib.predictors.ensemble import Ensemble
from naslib.optimizers.core.metaclasses import MetaOptimizer
from naslib.search_spaces.core.query_metrics import Metric


class SuccessiveHalving(MetaOptimizer):
    """
    Optimizer is randomly sampling architectures and queries/trains on the corresponding fidelities.
    After that, models will be discarded according to eta.
    DONE: Implement training
    """
    using_step_function = False

    def __init__(
        self,
        config,
        weight_optimizer=torch.optim.SGD,
        loss_criteria=torch.nn.CrossEntropyLoss(),
        grad_clip=None,
    ):
        """
        Initialize a Successive Halving optimizer.

        Args:
            config
            weight_optimizer (torch.optim.Optimizer): The optimizer to
                train the (convolutional) weights.
            loss_criteria (TODO): The loss
            grad_clip (float): Where to clip the gradients (default None).
        """
        super(SuccessiveHalving, self).__init__()
        self.weight_optimizer = weight_optimizer
        self.loss = loss_criteria
        self.grad_clip = grad_clip
        self.budget_max = config.search.budget_max
        self.performance_metric = Metric.VAL_ACCURACY
        self.dataset = config.dataset
        self.end = False
        self.fidelity = config.search.min_fidelity
        self.number_archs = config.search.number_archs
        self.eta = config.search.eta
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.budget_type = config.search.budget_type  # is not for one query is overall
        self.fidelity_counter = 0
        self.sampled_archs = []
        self.history = torch.nn.ModuleList()
        self.end = False
        self.old_fidelity = 0
        self.method = config.search.method
        #right now only for testing 
        if self.method == "tpe":#
            self.ss_type= config.search_space
            self.encoding_type = config.search.encoding_type
            #self.p = config.search.p
            #self.percentile = config.search.percentile
            self.N_min = 5
        self.optimizer_stats = defaultdict(lambda: defaultdict(list))

    def adapt_search_space(self, search_space, scope=None, dataset_api=None):
        assert (
            search_space.QUERYABLE
        ), "Successsive Halving is currently only implemented for benchmarks."
        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.dataset_api = dataset_api
    

    def new_epoch(self):
        """
        Sample a new architecture to train.
        # TODO: with this kind of architekeur, in evaluation only the last fideltiy
        """

        #model = torch.nn.Module()  # hacky way to get arch and accuracy checkpointable
        #model.arch = self.search_space.clone()
        #TODO is num_init needed 
        if len(self.sampled_archs) < self.number_archs:
            #model.arch.sample_random_architecture(dataset_api=self.dataset_api) 
            model = self.sample(self.method)
            
        else:
            model = self.sampled_archs[self.fidelity_counter]

#             return(ranks < self.num_configs[self.stage])

        model.accuracy = model.arch.query(
            self.performance_metric,
            self.dataset,
            epoch=int(self.fidelity),
            dataset_api=self.dataset_api,
        )

        budget = (self.fidelity - self.old_fidelity) / self.budget_max
        # DONE: make query type secure
        if self.budget_type == 'time':
            # DONE: make dependent on performance_metric
            model.time = model.arch.query(  # TODO: this is the time for training from screatch.
                self.performance_metric,
                self.dataset,
                epoch=int(self.fidelity),
                dataset_api=self.dataset_api,
            )
            budget = model.time
        elif not(self.budget_type == "epoch"):
            raise NameError("budget time should be time or epoch")
        # TODO: make this more beautiful/more efficient
        # DONE: we may need to track of all ever sampled archs
        if len(self.sampled_archs) < self.number_archs:
            self.sampled_archs.append(model)
        else:
            self.sampled_archs[self.fidelity_counter] = model
        self.update_optimizer_stats()
        self.fidelity_counter += 1
        # DONE: fidelity is changed for new epoch, what make the wrong values in the dictonary
        self._update_history(model)
        if self.fidelity_counter == self.number_archs:
            self.old_fidelity = self.fidelity
            self.fidelity = math.floor(self.eta*self.fidelity)
            self.sampled_archs.sort(key=lambda model: model.accuracy, reverse=True)
            if self.fidelity > self.budget_max:
                self.end = True
            elif(math.floor(self.number_archs/self.eta)) != 0:
                self.sampled_archs = self.sampled_archs[0:math.floor(self.number_archs/self.eta)]

            else:
                self.end = True
                self.sampled_archs = [self.sampled_archs[0]]  # but maybe there maybe a different way
            self.number_archs = len(self.sampled_archs)
            self.fidelity_counter = 0
        # TODO: budget equals
        return budget
       
    def _update_history(self, child):
        if len(self.history) < 100:
            self.history.append(child)
        else:
            for i, p in enumerate(self.history):
                if child.accuracy > p.accuracy:
                    self.history[i] = child
                    break

    def get_final_architecture(self):
        """
        Returns the sampled architecture with the lowest validation error.
        """
        return max(self.sampled_archs, key=lambda x: x.accuracy).arch

    def train_statistics(self, report_incumbent=True):

        if report_incumbent:
            best_arch = self.get_final_architecture()
        else:
            best_arch = self.sampled_archs[-1].arch

        return (
            best_arch.query(
                Metric.TRAIN_ACCURACY, self.dataset, dataset_api=self.dataset_api
            ),
            best_arch.query(
                Metric.VAL_ACCURACY, self.dataset, dataset_api=self.dataset_api
            ),
            best_arch.query(
                Metric.TEST_ACCURACY, self.dataset, dataset_api=self.dataset_api
            ),
            best_arch.query(
                Metric.TRAIN_TIME, self.dataset, dataset_api=self.dataset_api
            ),
        )

    def update_optimizer_stats(self):
        """
        Updates statistics of optimizer to be able to create useful plots
        """
        arch = self.sampled_archs[self.fidelity_counter].arch
        arch_hash = hash(self.sampled_archs[self.fidelity_counter])
        # this dict contains metrics to save
        metrics = {
            "train_acc": Metric.TRAIN_ACCURACY,
            "val_acc": Metric.VAL_ACCURACY,
            "test_acc": Metric.TEST_ACCURACY,
            "train_time": Metric.TRAIN_TIME
        }
        for metric_name, metric in metrics.items():
            metric_value = arch.query(
                metric,
                self.dataset,
                dataset_api=self.dataset_api,
                epoch=int(self.fidelity)
            )
            self.optimizer_stats[arch_hash][metric_name].append(metric_value)
        self.optimizer_stats[arch_hash]['fidelity'].append(self.fidelity)

    def test_statistics(self):
        best_arch = self.get_final_architecture()
        return best_arch.query(Metric.RAW, self.dataset, dataset_api=self.dataset_api)

    def get_op_optimizer(self):
        return self.weight_optimizer
    
    def get_checkpointables(self):
        return {"model": self.history}

    
    def sample(self, method):
        if method == "random" or  len(self.sampled_archs) < self.N_min:
            model = torch.nn.Module()  # hacky way to get arch and accuracy checkpointable
            model.arch = self.search_space.clone()
            model.arch.sample_random_architecture(dataset_api=self.dataset_api) 
        else:         
            xtrain = [m.arch for m in self.sampled_archs]
            ytrain = [m.accuracy for m in self.sampled_archs]
            ensemble =   Ensemble(
                predictor_type= "tpe",  
                num_ensemble= 1,
                encoding_type= self.encoding_type,
                ss_type=  self.ss_type
                )
            train_error = ensemble.fit(xtrain,ytrain,self.fidelity)
        #sample_KDE()
        return model

 
"""
def get_config(self, budget):
        
            #Function to sample a new configuration
            #This function is called inside Hyperband to query a new configuration
            #Parameters:
            #-----------
            #budget: float
            #    the budget for which this configuration is scheduled
            #returns: config
            #    should return a valid configuration
        
        
        self.logger.debug('start sampling a new configuration.')
        

        sample = None
        info_dict = {}
        
        # If no model is available, sample from prior
        # also mix in a fraction of random configs
        if len(self.kde_models.keys()) == 0 or np.random.rand() < self.random_fraction:
            sample =  self.configspace.sample_configuration()
            info_dict['model_based_pick'] = False

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
                    sample = self.configspace.sample_configuration().get_dictionary()
                    info_dict['model_based_pick']  = False
                else:
                    self.logger.debug('best_vector: {}, {}, {}, {}'.format(best_vector, best, l(best_vector), g(best_vector)))
                    for i, hp_value in enumerate(best_vector):
                        if isinstance(
                            self.configspace.get_hyperparameter(
                                self.configspace.get_hyperparameter_by_idx(i)
                            ),
                            ConfigSpace.hyperparameters.CategoricalHyperparameter
                        ):
                            best_vector[i] = int(np.rint(best_vector[i]))
                    sample = ConfigSpace.Configuration(self.configspace, vector=best_vector).get_dictionary()
                    
                    try:
                        sample = ConfigSpace.util.deactivate_inactive_hyperparameters(
                                    configuration_space=self.configspace,
                                    configuration=sample
                                    )
                        info_dict['model_based_pick'] = True

                    except Exception as e:
                        self.logger.warning(("="*50 + "\n")*3 +\
                                "Error converting configuration:\n%s"%sample+\
                                "\n here is a traceback:" +\
                                traceback.format_exc())
                        raise(e)

            except:
                self.logger.warning("Sampling based optimization with %i samples failed\n %s \nUsing random configuration"%(self.num_samples, traceback.format_exc()))
                sample = self.configspace.sample_configuration()
                info_dict['model_based_pick']  = False


        try:
            sample = ConfigSpace.util.deactivate_inactive_hyperparameters(
                configuration_space=self.configspace,
                configuration=sample.get_dictionary()
            ).get_dictionary()
        except Exception as e:
            self.logger.warning("Error (%s) converting configuration: %s -> "
                                "using random configuration!",
                                e,
                                sample)
            sample = self.configspace.sample_configuration().get_dictionary()
        self.logger.debug('done sampling a new configuration.')
        return sample, info_dict
"""
