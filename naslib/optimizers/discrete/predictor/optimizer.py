import collections
import logging
import torch
import copy
import numpy as np

from naslib.optimizers.core.metaclasses import MetaOptimizer
from naslib.optimizers.discrete.utils.utils import sample_random_architecture
from naslib.optimizers.discrete.utils.encodings import encode
from naslib.optimizers.discrete.predictor.predictors import Ensemble

from naslib.search_spaces.core.query_metrics import Metric

from naslib.utils.utils import AttrDict, count_parameters_in_MB
from naslib.utils.logging import log_every_n_seconds


logger = logging.getLogger(__name__)


class Predictor(MetaOptimizer):
    
    # training the models is not implemented
    using_step_function = False
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.epochs = config.search.epochs

        self.performance_metric = Metric.VAL_ACCURACY
        self.dataset = config.dataset

        # 20, 172 are the magic numbers from [Wen et al. 2019]
        self.k = int(20 / 172 * self.epochs)
        self.num_init = self.epochs - self.k
        self.test_size = 2 * self.epochs
        
        self.predictor_type = 'feedforward'
        self.num_ensemble = 3
        self.debug_predictor = True
        
        self.train_data = []
        self.choices = []        
        self.history = torch.nn.ModuleList()


    def adapt_search_space(self, search_space, scope=None):
        assert search_space.QUERYABLE, "Regularized evolution is currently only implemented for benchmarks."
        
        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
    
    def new_epoch(self, epoch):

        if epoch < self.num_init:
            logger.info("Start sampling architectures to generate training data")
            # If there is no scope defined, let's use the search space default one
            
            model = torch.nn.Module()   # hacky way to get arch and accuracy checkpointable
            
            model.arch = sample_random_architecture(self.search_space, self.scope)
            model.accuracy = model.arch.query(self.performance_metric, self.dataset)
            
            self.train_data.append(model)
            self._update_history(model)

        else:
            if epoch == self.num_init:
                # train the neural predictor and use it to predict arches in test_data
                
                test_data = []
                for i in range(self.test_size):
                    model = torch.nn.Module()
                    model.arch = sample_random_architecture(self.search_space, self.scope)
                    test_data.append(model)
                                        
                xtrain = [encode(m.arch) for m in self.train_data]
                ytrain = [m.accuracy for m in self.train_data]
                xtest = [encode(m.arch) for m in test_data]
                
                predictor = Ensemble(predictor_type=self.predictor_type,
                                     num_ensemble=self.num_ensemble)
                predictor.fit(xtrain, ytrain)                
                train_pred = np.squeeze(predictor.predict(xtrain))
                test_pred = np.squeeze(predictor.predict(xtest))  

                if self.num_ensemble > 1:
                    train_pred = np.mean(train_pred, axis=0)
                    test_pred = np.mean(test_pred, axis=0)

                if self.debug_predictor:
                    self.evaluate_predictor(xtrain=xtrain, 
                                            ytrain=ytrain, 
                                            xtest=xtest, 
                                            train_pred=train_pred,
                                            test_pred=test_pred,
                                            test_data=test_data)
                
                sorted_indices = np.argsort(test_pred)[-self.k:]
                for i in sorted_indices:
                    self.choices.append(test_data[i].arch)                

            # train the next chosen architecture
            choice = torch.nn.Module()   # hacky way to get arch and accuracy checkpointable
            choice.arch = self.choices[epoch - self.num_init]
            choice.accuracy = choice.arch.query(self.performance_metric, self.dataset)
            self._update_history(choice)
        
    def evaluate_predictor(self, xtrain, 
                           ytrain, 
                           xtest, 
                           train_pred, 
                           test_pred, 
                           test_data, 
                           slice_size=4):
        """
        This method is only used for debugging purposes.
        Query the architectures in the set so that we can evaluate
        the performance of the predictor.
        """
        ytest = []
        for model in test_data:
            ytest.append(model.arch.query(self.performance_metric, self.dataset))

        print('ytrain shape', np.array(ytrain).shape)
        print('train_pred shape', np.array(train_pred).shape)
        print('ytest shape', np.array(ytest).shape)
        print('test_pred shape', np.array(test_pred).shape)
        train_error = np.mean(abs(train_pred-ytrain))
        test_error = np.mean(abs(test_pred-ytest))
        correlation = np.corrcoef(np.array(ytest), np.array(test_pred))[1,0]
        print('train error', train_error)
        print('test error', test_error)
        print('correlation', correlation)
        print()
        print('xtrain slice', xtrain[:slice_size])                
        print('ytrain slice', ytrain[:slice_size])                
        print('train_pred slice', train_pred[:slice_size])
        print()
        print('xtest slice', xtest[:slice_size])
        print('ytest slice', ytest[:slice_size])
        print('test_pred slice', test_pred[:slice_size])

    def _update_history(self, child):
        if len(self.history) < 100:
            self.history.append(child)
        else:
            for i, p in enumerate(self.history):
                if child.accuracy > p.accuracy:
                    self.history[i] = child
                    break

    def train_statistics(self):
        best_arch = self.get_final_architecture()
        return (
            best_arch.query(Metric.TRAIN_ACCURACY, self.dataset), 
            best_arch.query(Metric.TRAIN_LOSS, self.dataset), 
            best_arch.query(Metric.VAL_ACCURACY, self.dataset), 
            best_arch.query(Metric.VAL_LOSS, self.dataset), 
        )
    

    def test_statistics(self):
        best_arch = self.get_final_architecture()
        return best_arch.query(Metric.RAW, self.dataset)


    def get_final_architecture(self):
        return max(self.history, key=lambda x: x.accuracy).arch
    
    
    def get_op_optimizer(self):
        raise NotImplementedError()

    
    def get_checkpointables(self):
        return {'model': self.history}
    

    def get_model_size(self):
        return count_parameters_in_MB(self.history)