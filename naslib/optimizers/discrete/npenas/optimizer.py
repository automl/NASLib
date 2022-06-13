import collections
import logging
import torch
import copy
import random
import numpy as np

from naslib.optimizers.core.metaclasses import MetaOptimizer
from naslib.optimizers.discrete.bananas.acquisition_functions import (
    acquisition_function,
)

from naslib.predictors.ensemble import Ensemble
from naslib.predictors import ZeroCost
from naslib.predictors.utils.encodings import encode_spec


from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.nasbench101.conversions import convert_tuple_to_spec, convert_spec_to_tuple


from naslib.utils.utils import AttrDict, count_parameters_in_MB, get_train_val_loaders
from naslib.utils.logging import log_every_n_seconds


logger = logging.getLogger(__name__)


class Npenas(MetaOptimizer):

    # training the models is not implemented
    using_step_function = False

    def __init__(self, config, zc_api):
        super().__init__()
        self.config = config
        self.epochs = config.search.epochs

        self.performance_metric = Metric.VAL_ACCURACY
        self.dataset = config.dataset

        self.k = config.search.k
        self.num_init = config.search.num_init
        self.num_ensemble = config.search.num_ensemble
        self.predictor_type = config.search.predictor_type
        #missing acq fn type (its) and acq fn optimization (random sampling)
        self.encoding_type = config.search.encoding_type  # currently not implemented
        self.num_arches_to_mutate = config.search.num_arches_to_mutate
        self.max_mutations = config.search.max_mutations
        self.num_candidates = config.search.num_candidates
        self.max_zerocost = 1000

        self.train_data = []
        self.next_batch = []
        self.history = torch.nn.ModuleList()

        #self.zc = "omni" in self.predictor_type
        #self.semi = "semi" in self.predictor_type

        self.zc = config.search.zc_ensemble
        self.zc_names = config.search.zc_names

        self.sample_from_zc_api = zc_api is not None
        self.zc_api = zc_api

    def adapt_search_space(self, search_space, scope=None, dataset_api=None):
        assert (
            search_space.QUERYABLE
        ), "Npenas is currently only implemented for benchmarks."

        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.dataset_api = dataset_api
        self.ss_type = self.search_space.get_type()
        if self.zc:
            self.train_loader, _, _, _, _ = get_train_val_loaders(
                self.config
            )

    def get_zero_cost_predictors(self):
        return [ZeroCost(method_type=zc_name) for zc_name in self.zc_names]

    def query_zc_scores(self, arch, predictors, zc_api):
        zc_scores = {}

        for predictor in predictors:
            score = zc_api[str(arch)][predictor]['score']

            if float("-inf") == score:
                score = -1e9
            elif float("inf") == score:
                score = 1e9

            zc_scores[predictor] = score

        return zc_scores


    def new_epoch(self, epoch):

        if epoch < self.num_init:
            # randomly sample initial architectures
            model = torch.nn.Module()
            #model.ss = self.search_space.clone()
            self.search_space.sample_random_architecture(dataset_api=self.dataset_api, load_labeled=self.sample_from_zc_api)

            model.arch_hash = self.search_space.get_hash()
            model.arch = encode_spec(model.arch_hash, encoding_type='adjacency_one_hot', ss_type=self.search_space.get_type())
            model.accuracy = self.zc_api[str(model.arch_hash)]['val_accuracy']

            if self.zc and len(self.train_data) <= self.max_zerocost:
                model.zc_scores = self.query_zc_scores(model.arch_hash, self.zc_names, self.zc_api)

            self.train_data.append(model)
            self._update_history(model)

        else:
            if len(self.next_batch) == 0:
                # train a neural predictor
                xtrain = [m.arch for m in self.train_data]
                ytrain = [m.accuracy for m in self.train_data]
                ensemble = Ensemble(
                    num_ensemble=self.num_ensemble,
                    ss_type=self.ss_type,
                    predictor_type=self.predictor_type,
                    zc=self.zc,
                    zc_only=self.config.search.zc_only,
                    config=self.config,
                )

                if self.zc and len(self.train_data) <= self.max_zerocost:
                    # pass the zero-cost scores to the predictor
                    train_info = {'zero_cost_scores': [m.zc_scores for m in self.train_data]}
                    ensemble.set_pre_computations(xtrain_zc_info=train_info)

                train_error = ensemble.fit(xtrain, ytrain)

                # define an acquisition function
                acq_fn = acquisition_function(
                    ensemble=ensemble, ytrain=None, acq_fn_type="exploit_only"
                )

                # optimize the acquisition function to output k new architectures
                candidates = []
                zc_scores = []

                # mutate the k best architectures by x
                best_arch_indices = np.argsort(ytrain)[-self.num_arches_to_mutate :]
                best_arches = [self.train_data[i].arch_hash for i in best_arch_indices]
                candidates = []
                for arch in best_arches:
                    for _ in range(
                        int(self.num_candidates / len(best_arches) / self.max_mutations)
                    ):
                        candidate = torch.nn.Module()
                        current_hash = copy.deepcopy(arch)
                        for edit in range(int(self.max_mutations)):
                            '''
                            arch = self.search_space.clone()
                            arch.mutate(candidate, dataset_api=self.dataset_api)
                            candidate = arch
                            '''
                            new_arch_hash = self.mutate_arch(current_hash, dataset_api=self.dataset_api)
                            current_hash = new_arch_hash

                        candidate.arch_hash = current_hash
                        candidate.arch = encode_spec(candidate.arch_hash, encoding_type='adjacency_one_hot', ss_type=self.search_space.get_type())
                        candidate.accuracy = self.zc_api[str(candidate.arch_hash)]['val_accuracy']
                        candidates.append(candidate)

                if self.zc:
                    for model in candidates:
                        model.zc_scores = self.query_zc_scores(model.arch_hash, self.zc_names, self.zc_api)

                    values = [acq_fn(model.arch, [{'zero_cost_scores' : model.zc_scores}]) for model in candidates]
                else:
                    values = [acq_fn(model.arch) for model in candidates]
                sorted_indices = np.argsort(values)
                choices = [candidates[i] for i in sorted_indices[-self.k :]]
                self.next_batch = [*choices]

            # train the next architecture chosen by the neural predictor
            model = self.next_batch.pop()
            model.accuracy = self.zc_api[str(model.arch_hash)]['val_accuracy']
            if self.zc and len(self.train_data) <= self.max_zerocost:
                model.zc_scores = self.query_zc_scores(model.arch_hash, self.zc_names, self.zc_api)


            self._update_history(model)
            self.train_data.append(model)

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
        '''
        if report_incumbent:
            best_arch = self.get_final_architecture()
                else:
            best_arch = self.train_data[-1].arch
        
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
        '''
        return (
            -1,
            self.zc_api[str(best_arch)]['val_accuracy'],
            -1,
            -1,
        )

    def mutate_arch(self, arch_hash, dataset_api):
        if self.search_space.get_type() == 'nasbench201':
            OP_NAMES = ["Identity", "Zero", "ReLUConvBN3x3", "ReLUConvBN1x1", "AvgPool1x1"]

            op_indices = list(arch_hash)
            edge = np.random.choice(len(arch_hash))
            available = [o for o in range(len(OP_NAMES)) if o != arch_hash[edge]]
            op_index = np.random.choice(available)
            op_indices[edge] = op_index

            return tuple(op_indices)

        elif self.search_space.get_type() == 'nasbench101':
            NUM_VERTICES = 7
            CONV1X1 = 'conv1x1-bn-relu'
            CONV3X3 = 'conv3x3-bn-relu'
            MAXPOOL3X3 = 'maxpool3x3'
            OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]

            arch_spec = convert_tuple_to_spec(arch_hash)
            spec = copy.deepcopy(arch_spec)
            matrix, ops = spec['matrix'], spec['ops']
            for _ in range(1):
                while True:
                    new_matrix = copy.deepcopy(matrix)
                    new_ops = copy.deepcopy(ops)
                    for src in range(0, NUM_VERTICES - 1):
                        for dst in range(src+1, NUM_VERTICES):
                            if np.random.random() < 1 / NUM_VERTICES:
                                new_matrix[src][dst] = 1 - new_matrix[src][dst]
                    for ind in range(1, NUM_VERTICES - 1):
                        if np.random.random() < 1 / len(OPS):
                            available = [op for op in OPS if op != new_ops[ind]]
                            new_ops[ind] = np.random.choice(available)

                    new_spec = dataset_api['api'].ModelSpec(new_matrix, new_ops)
                    if dataset_api['nb101_data'].is_valid(new_spec):
                        break

            return convert_spec_to_tuple({"matrix": new_matrix, "ops": new_ops})

        elif self.search_space.get_type() == 'nasbench301':
            #mutate function doesn't exist? search space is too big
            raise NotImplementedError()

        elif self.search_space.get_type() == 'transbench101_micro':
            OP_NAMES = ['Identity', 'Zero', 'ReLUConvBN3x3', 'ReLUConvBN1x1']

            op_indices = list(arch_hash)
            edge = np.random.choice(len(arch_hash))
            available = [o for o in range(len(OP_NAMES)) if o != arch_hash[edge]]
            op_index = np.random.choice(available)
            op_indices[edge] = op_index

            return tuple(op_indices)

        elif self.search_space.get_type() == 'transbench101_macro':
            parent_op_indices = list(arch_hash)
            #parent_op_ind = [ind for ind in parent_op_indices if ind]
            def f(g):
                r = len(g)
                p = sum([int(i==4 or i==3) for i in g])
                q = sum([int(i==4 or i==2) for i in g])
                return r, p, q

            def g(r, p, q):
                u = [2*int(i<p) for i in range(r)]
                v = [int(i<q) for i in range(r)]
                w = [1+sum(x) for x in zip(u, v)]
                return np.random.permutation(w)

            op_indices = []
            while len(op_indices)<6:
                a, b, c = f(parent_op_indices)

                a_available = [i for i in [4, 5, 6] if i!=a]
                b_available = [i for i in range(1, 5) if i!=b]
                c_available = [i for i in range(1, 4) if i!=c]
                
                dic1 = {1: a, 2: b, 3: c}
                dic2 = {1: a_available, 2: b_available, 3: c_available}
                
                numb = random.randint(1, 3)
                
                dic1[numb] = random.choice(dic2[numb])

                op_indices = g(dic1[1], dic1[2], dic1[3])
            #while len(op_indices)<6:
            #    op_indices = np.append(op_indices, 0)

            return tuple(op_indices)

        else:
            raise NotImplementedError()

    def test_statistics(self):
        #best_arch = self.get_final_architecture()
        #return best_arch.query(Metric.RAW, self.dataset, dataset_api=self.dataset_api)
        return {}

    def get_final_architecture(self):
        return max(self.history, key=lambda x: x.accuracy).arch_hash

    def get_op_optimizer(self):
        raise NotImplementedError()

    def get_checkpointables(self):
        return {"model": self.history}

    def get_model_size(self):
        return count_parameters_in_MB(self.history)

    def get_arch_as_string(self, arch):
        if self.search_space.get_type() == 'nasbench301':
            str_arch = str(list((list(arch[0]), list(arch[1]))))
        else:
            str_arch = str(arch)
        return str_arch