import os
import pickle
import numpy as np
import copy
import random
import torch

from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.core.graph import Graph
from naslib.utils.utils import get_project_root
from naslib.search_spaces.nasbenchasr.conversions import flatten, copy_structure


OP_NAMES = ['linear', 'conv5', 'conv5d2', 'conv7', 'conv7d2', 'zero']


class NasBenchASRSearchSpace(Graph):
    """
    Contains the interface to the tabular benchmark of nas-bench-asr.
    Note: currently we do not support building a naslib object for
    nas-bench-asr architectures.
    """

    QUERYABLE = True

    def __init__(self):
        super().__init__()
        self.load_labeled = False
        self.max_epoch = 40
        self.max_nodes = 3
        self.accs = None
        self.compact = None


    def query(self, metric=None, dataset=None, path=None, epoch=-1,
              full_lc=False, dataset_api=None):
        """
        Query results from nas-bench-asr
        """
        metric_to_asr = {
            Metric.VAL_ACCURACY: "val_per",
            Metric.TEST_ACCURACY: "test_per",
            Metric.PARAMETERS: "params",
            Metric.FLOPS: "flops",
        }

        assert self.compact is not None
        assert metric in [
            Metric.TRAIN_ACCURACY,
            Metric.TRAIN_LOSS,
            Metric.VAL_ACCURACY,
            Metric.TEST_ACCURACY,
            Metric.PARAMETERS,
            Metric.FLOPS,
            Metric.TRAIN_TIME,
            Metric.RAW,
        ]
        # seed has to be set manually, otherwise one would get different metric values for the same architecture!
        query_results = dataset_api["asr_data"].full_info(self.compact, seed=1234)


        if metric != Metric.VAL_ACCURACY:
            if metric == Metric.TEST_ACCURACY:
                return query_results[metric_to_asr[metric]]
            elif (metric == Metric.PARAMETERS) or (metric == Metric.FLOPS):
                return query_results['info'][metric_to_asr[metric]]
            elif (metric == Metric.TRAIN_TIME):
                if epoch == -1:
                    return self.get_max_epochs() + 1
                else:
                    return 1 * epoch
            elif metric in [Metric.TRAIN_ACCURACY, Metric.TRAIN_LOSS,
                            Metric.TRAIN_TIME, Metric.RAW]:
                return -1
        else:
            if full_lc and epoch == -1:
                return [
                    loss for loss in query_results[metric_to_asr[metric]]
                ]
            elif full_lc and epoch != -1:
                return [
                    loss for loss in query_results[metric_to_asr[metric]][:epoch]
                ]
            else:
                # return the value of the metric only at the specified epoch
                return float(query_results[metric_to_asr[metric]][epoch])

    def get_compact(self):
        assert self.compact is not None
        return self.compact

    def get_hash(self):
        return self.get_compact()

    def set_compact(self, compact):
        self.compact = compact

    def sample_random_architecture(self, dataset_api):
        search_space = [[len(OP_NAMES)] + [2]*(idx+1) for idx in
                        range(self.max_nodes)]
        flat = flatten(search_space)
        m = [random.randrange(opts) for opts in flat]
        m = copy_structure(m, search_space)

        compact = m
        self.set_compact(compact)
        return compact
    def model_based_sample_architecture(self, dataset_api=None, minimize_me=None, good_kde=None, vartypes=None):
        """
        This will perform a model-based architecture sampling and update the edges in the
        naslib object accordingly.
        """
        num_samples = 128
        random_fraction = 0.33
        best = np.inf
        best_vector = None
        for i in range(num_samples):
            idx = np.random.randint(0, len(good_kde.data))
            datum = good_kde.data[idx]
            vector = []
            for m, bw, t in zip(datum, good_kde.bw, vartypes):
                if np.random.rand() < (1 - bw):
                    vector.append(int(m))
                else:
                    vector.append(np.random.randint(t))
            val = minimize_me(vector)
            if val < best:
                best = val
                best_vector = vector
        if best_vector is None or np.random.rand() < random_fraction:
            self.sample_random_architecture(dataset_api=dataset_api)
        else:
            for i in range(len(best_vector)):
                best_vector[i] = int(np.rint(best_vector[i]))
            best_vector = [[best_vector[0],best_vector[1]],
                           [best_vector[2],best_vector[3],best_vector[4]],
                           [best_vector[5],best_vector[6], best_vector[7], best_vector[8]]]
            self.set_compact(best_vector)


    def mutate(self, parent, mutation_rate=1, dataset_api=None):
        """
        This will mutate the cell in one of two ways:
        change an edge; change an op.
        Todo: mutate by adding/removing nodes.
        Todo: mutate the list of hidden nodes.
        Todo: edges between initial hidden nodes are not mutated.
        """
        parent_compact = parent.get_compact()
        compact = copy.deepcopy(parent_compact)

        for _ in range(int(mutation_rate)):
            mutation_type = np.random.choice([2])
           
            if mutation_type == 1:
                # change an edge
                # first pick up a node
                node_id = np.random.choice(3)
                node = compact[node_id]
                # pick up an edge id
                edge_id = np.random.choice(len(node[1:])) + 1
                # edge ops are in [identity, zero] ([0, 1])
                new_edge_op = int(not compact[node_id][edge_id])
                # apply the mutation
                compact[node_id][edge_id] = new_edge_op

            elif mutation_type == 2:
                # change an op
                node_id = np.random.choice(3)
                node = compact[node_id]
                op_id = node[0]
                list_of_ops_ids = list(range(len(OP_NAMES)))
                list_of_ops_ids.remove(op_id)
                new_op_id = random.choice(list_of_ops_ids)
                compact[node_id][0] = new_op_id

        self.set_compact(compact)
    def crossover_bin(self, parent, mutant, dim, prob, dataset_api=None):
        '''Performs the binomial crossover of DE
           this write now only for the things that mutate 
        '''
        parent_compact = parent.get_compact()
        compact = copy.deepcopy(parent_compact)
        mutant_compact= mutant.get_compact()
        cross_node = np.random.rand(3) < prob
        cross_type = np.random.rand(2) < prob
        for i,node in enumerate(cross_node):
            if not node:
                continue
            if cross_type[0]:
                cross_edges = np.random.rand(len(compact[i][1:])) 
                for j,egde in enumerate(cross_edges):
                    if not egde:
                        continue
                    compact[i][j+1] =  mutant_compact[i][j+1]
            if cross_type[1]:
                compact[i][0] = mutant_compact[i][0]
        self.set_compact(compact)

      

    def get_nbhd(self, dataset_api=None):
        """
        Return all neighbors of the architecture
        """
        compact = self.get_compact()
        #edges, ops, hiddens = compact
        nbhd = []

        def add_to_nbhd(new_compact, nbhd):
            print(new_compact)
            nbr = NasBenchASRSearchSpace()
            nbr.set_compact(new_compact)
            nbr_model = torch.nn.Module()
            nbr_model.arch = nbr
            nbhd.append(nbr_model)
            return nbhd

        for node_id in range(len(compact)):
            node = compact[node_id]
            for edge_id in range(len(node)):
                if edge_id == 0:
                    edge_op = compact[node_id][0]
                    list_of_ops_ids = list(range(len(OP_NAMES)))
                    list_of_ops_ids.remove(edge_op)
                    for op_id in list_of_ops_ids:
                        new_compact = copy.deepcopy(compact)
                        new_compact[node_id][0] = op_id
                        nbhd = add_to_nbhd(new_compact, nbhd)
                else:
                    edge_op = compact[node_id][edge_id]
                    new_edge_op = int(not edge_op)
                    new_compact = copy.deepcopy(compact)
                    new_compact[node_id][edge_id] = new_edge_op
                    nbhd = add_to_nbhd(new_compact, nbhd)

        random.shuffle(nbhd)
        return nbhd

    def get_type(self):
        return 'asr'

    def get_max_epochs(self):
        return 39

