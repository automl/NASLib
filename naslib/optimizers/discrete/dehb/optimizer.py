from logging import raiseExceptions
import numpy as np
import torch
import copy
import collections
import math
from collections import defaultdict
from naslib.optimizers import SuccessiveHalving as SH
from naslib.optimizers.core.metaclasses import MetaOptimizer
from naslib.search_spaces.core.query_metrics import Metric

from collections import defaultdict


class DEHB(MetaOptimizer):
    """
    This is a Hyperband Implementation, that uses the Sucessive Halving Algorithm with different settings.
    """

    # training the models is not implemented
    using_step_function = False

    def __init__(
        self,
        config,
        weight_optimizer=torch.optim.SGD,
        loss_criteria=torch.nn.CrossEntropyLoss(),
        grad_clip=None,
    ):
        """
        Initialize a Successive Halving  optimizer.

        Args:
            config
            weight_optimizer (torch.optim.Optimizer): The optimizer to
                train the (convolutional) weights.
            loss_criteria (TODO): The loss
            grad_clip (float): Where to clip the gradients (default None).
        """
        super(DEHB, self).__init__()
        # DE related variables
        self.weight_optimizer = weight_optimizer
        self.loss = loss_criteria
        self.grad_clip = grad_clip
        self.performance_metric = Metric.VAL_ACCURACY
        self.dataset = config.dataset
        self.fidelity = config.search.min_fidelity
        self.number_archs = config.search.number_archs
        self.eta = config.search.eta
        self.budget_max = config.search.budget_max 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.budget_type = config.search.budget_type  # is not for one query is overall
        self.fidelity_counter = 0
        self.history = torch.nn.ModuleList()
        self.end = False
        self.old_fidelity = 0 
        self.round_number = 0
        self.top_n_percent = 0.3
        self.method = config.search.method
        self.s_max = math.floor(math.log(self.budget_max, self.eta))
        self.b = (self.s_max + 1)* self.budget_max
        self.s = self.s_max
        self.current_round = []
        self.next_round = []
        self.new_fidelity = False # TODO think of more ellegant sollution
        self.de = dict()
        self._epsilon = 1e-6
        self.pop_size = {}       
        self.de[self.fidelity] = {}
        self.enc_dim = 6
        self.max_mutations = 1
        self.crossover_prob = 0.5
        self.top_n_percent = 0.3
        self.mutate_prob = 0.1
        self.de[self.fidelity]['promotions'] = collections.deque(maxlen=100)
        # Global trackers
        self.population = None
        self.fitness = None
        self.inc_score = np.inf
        self.inc_config = None
       
        
    def adapt_search_space(self, search_space, scope=None, dataset_api=None):
        assert (
            search_space.QUERYABLE
        ), "Successsive Halving is currently only implemented for benchmarks."
        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.dataset_api = dataset_api
        #self.max_training_epoch = self.search_space.get_max_epochs() #deifne
        self.max_training_epoch = 200
       
    def new_epoch(self):   #TODO: epoch, round and i have to be adappted 
        #calculate s n,r not before but during like in bohb 
        #
        ##this mind help if we need to parraleise 
        #if self.process < i: # re-init for each new process
        #    self.current_round = []
        #    self.next_round = []
        #    self.round_number = 0
        #    self.prev_round = 0
        #    self.counter = 0
        #    self.process = i
       
        print("bracket: {}, number_archs: {}, fidelity:{}, counter: {}".format(self.s,self.number_archs, self.fidelity, self.fidelity_counter))
        if self.round_number >= self.s: # reset round_number for each new round
            self.s -=1 
            self.round_number = 0
            n = math.ceil(self.b / self.budget_max / (self.s + 1) * self.eta ** self.s)
            r = self.budget_max * self.eta ** (-self.s)
            self.number_archs = n
            self.fidelity  = r
            print("bracket: {}, number_archs: {}, fidelity:{}".format(self.s,self.number_archs, self.fidelity))
        if self.s < 1:
            raise NameError("this method is finish")


        if self.fidelity_counter  <  self.number_archs:
            # sample random architectures
            model = torch.nn.Module()   # hacky way to get arch and accuracy checkpointable
            model.arch = self.search_space.clone()
            budget = self.fidelity
            if self.s == self.s_max:
                model.arch.sample_random_architecture(dataset_api=self.dataset_api) #sample in the first round 
            else:
                if len(self.de[self.fidelity]['promotions']) > 0:  #if we have allready something mutate the best 
                    print('promotion from budget: {}, length: {}'.format(self.fidelity, len(self.de[self.fidelity]['promotions'])))
                    model = self.de[self.fidelity]['promotions'].pop()
                    model = copy.deepcopy(model)
                    arch = self.search_space.clone()
                    arch.mutate(model.arch, dataset_api=self.dataset_api)
                    model.arch = arch
                else:
                    model.arch.sample_random_architecture(dataset_api=self.dataset_api) #if not do something random
            #model.epoch = self.fidelity # has to be changed
            # TODO: this is really good idea to implemtent it in all optimizer (that we write or in other but not from us) 
            model.epoch = int(min(self.fidelity, self.max_training_epoch)) #
            model.accuracy = model.arch.query(self.performance_metric,
                                              self.dataset,
                                              epoch=model.epoch,
                                              dataset_api=self.dataset_api)
            budget = (self.fidelity - self.old_fidelity) / self.budget_max
            # DONE: make query type secure
            if self.budget_type == 'time':
            #TODO also for predictions
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
            #Maybe this needs to the beginnig of this if 
            self.fidelity_counter += 1

            self._update_history(model)
            self.next_round.append(model)

        else:
           
            #self.new_fidelity = True
            if len(self.current_round) == 0: 
                self.new_fidelity = False
                self.old_fidelity = self.fidelity
                self.fidelity = math.floor(self.eta*self.fidelity)
                self.number_archs = math.floor(self.number_archs/self.eta) #maybe +1 or -1 not quite sure 
            
                self.de[self.fidelity] = {}
                self.de[self.fidelity]['promotions'] = collections.deque(maxlen=100)
                self.round_number += 1
                #n = math.ceil(int(self.b / self.budget_max / (self.s + 1)) * self.eta ** self.s)
                #r = self.budget_max * self.eta ** (-self.s)
                #self.number_archs = n
                #self.fidelity  = r
                self.fidelity_counter = 0
                self.de[self.fidelity] = {}
                self.de[self.fidelity]['promotions'] = collections.deque(maxlen=100)
                # if we are at the end of a round of hyperband, continue training only the best
                print("Starting a new round: continuing to train the best arches")
                cutoff = math.ceil(self.number_archs * self.top_n_percent)
                self.current_round = sorted(self.next_round, key=lambda x: -x.accuracy)[:cutoff]
                print("bracket: {}, number_archs: {}, fidelity:{}".format(self.s,self.number_archs, self.fidelity))
                #self.round_number = min(self.round_number, len(self.fidelities[round]) - 1)
                self.next_round = []
        
                
            
              
               
            # train the next architecture
            model = self.current_round.pop()
            self.fidelity_counter += 1
            """
            Note: technically we would just continue training this arch, but right now,
            just for simplicity, we treat it as if we start to train it again from scratch
            """
            model = copy.deepcopy(model)

            if np.random.rand(1) < self.mutate_prob:  #mutate through any     
                candidate = model.arch.clone()
                for _ in range(self.max_mutations): #here the mutation happens.
                    arch_ = self.search_space.clone()
                    arch_.mutate(candidate, dataset_api=self.dataset_api)
                    candidate = arch_
                mutant = candidate
                arch = self.search_space.clone()
                arch.crossover_bin(model.arch, mutant, self.enc_dim, self.crossover_prob, dataset_api=self.dataset_api) #this needs to be implemeted 
                model.arch = arch
            #model.epoch = self.fidelity
            model.epoch = int(min(self.fidelity, self.max_training_epoch))
            model.accuracy = model.arch.query(self.performance_metric,
                                              self.dataset,
                                              epoch=model.epoch,
                                              dataset_api=self.dataset_api)
            budget = (self.fidelity - self.old_fidelity) / self.budget_max
            # DONE: make query type secure
            if self.budget_type == 'time':
            #TODO also for predictions
            # DONE: make dependent on performance_metric
                model.time = model.arch.query(  # TODO: this is the time for training from screatch.
                self.performance_metric,
                self.dataset,
                epoch=model.epoch,
                dataset_api=self.dataset_api,
                )
                budget = model.time
            elif not(self.budget_type == "epoch"):
                raise NameError("budget time should be time or epoch")
            self.de[self.fidelity]['promotions'].append(model)  #apened model if round is finsih 
            self._update_history(model)
            self.next_round.append(model)
        return budget
    def _update_history(self, child): #not even sure if this is needed, but why not 
        self.history.append(child)

    def get_final_architecture(self):
        """
        Returns the sampled architecture with the lowest validation error.
        """
        best_arch = max(self.history, key=lambda x: x.accuracy)
        return best_arch.arch

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


    

    def train_model_statistics(self, report_incumbent=True):

        
        best_arch = self.sampled_archs[self.fidelity_counter -1].arch
        best_arch_hash = hash(self.sampled_archs[self.fidelity_counter -1])
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
            self.fidelity,
            best_arch_hash,
        )
    
    def update_optimizer_stats(self):
        """
        Updates statistics of optimizer to be able to create useful plots
        TODO i have to expand the dictionary such that we keep track of all parallel sh evaluations
        """
        self.optimizer_stats[self.s] = self.sh.optimizer_stats



    def test_statistics(self):
        return False
        best_arch = self.get_final_architecture()
        return best_arch.query(Metric.RAW, self.dataset, dataset_api=self.dataset_api)

    def get_op_optimizer(self):
        return self.weight_optimizer

    def get_checkpointables(self):
        return {"model": self.history}
