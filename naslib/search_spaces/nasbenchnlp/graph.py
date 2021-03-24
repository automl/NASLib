import os
import pickle
import numpy as np
import copy
import random

from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.core.graph import Graph
from naslib.utils.utils import get_project_root


class NasBenchNLPSearchSpace(Graph):
    """
    Contains the interface to the tabular benchmark of nas-bench-nlp.
    Note: currently we do not support building a naslib object for 
    nas-bench-nlp architectures.
    """

    QUERYABLE = False

    def __init__(self):
        super().__init__()
        
    def load_labeled_architecture(self, dataset_api=None, max_nodes=25):
        """
        This is meant to be called by a new NasBenchNLPSearchSpace() object.
        It samples a random architecture from the nas-bench-nlp data.
        """
        while True:
            index = np.random.choice(len(dataset_api['nlp_arches']))
            compact = dataset_api['nlp_arches'][index]
            if len(compact[1]) <= max_nodes:
                break
        self.load_labeled = True
        self.set_compact(compact)

    def query(self, metric=None, dataset=None, path=None, epoch=-1, full_lc=False, dataset_api=None):
        """
        Query results from nas-bench-nlp
        """
        metric_to_nlp = {
            Metric.TRAIN_ACCURACY: 'train_losses',
            Metric.VAL_ACCURACY: 'val_losses',
            Metric.TEST_ACCURACY: 'test_losses',
            Metric.TRAIN_TIME: 'wall_times',
            Metric.TRAIN_LOSS: 'train_losses',

        }
        
        assert self.load_labeled
        """
        If we loaded the architecture from the nas-bench-nlp data (using 
        load_labeled_architecture()), then self.compact will contain the architecture spec.
        """
        try:
            assert metric in [Metric.TRAIN_ACCURACY, Metric.TRAIN_LOSS, Metric.VAL_ACCURACY, \
                          Metric.TEST_ACCURACY, Metric.TRAIN_TIME]
        except:
            print('hold')
        query_results = dataset_api['nlp_data'][self.compact]

        sign = 1
        if metric == Metric.TRAIN_LOSS:
            sign = -1
        
        if metric == Metric.TRAIN_TIME:
            return query_results[metric_to_nlp[metric]]
        elif metric == Metric.HP:
            # todo: compute flops/params/latency for each arch
            return {'flops': 15, 'params': 0.1, 'latency': 0.01}
        elif full_lc and epoch == -1:
            return [sign * (100 - loss) for loss in query_results[metric_to_nlp[metric]]]
        elif full_lc and epoch != -1:
            return [sign * (100 - loss) for loss in query_results[metric_to_nlp[metric]][:epoch]]
        else:
            # return the value of the metric only at the specified epoch
            return sign * (100 - query_results[metric_to_nlp[metric]][epoch])

    def get_compact(self):
        assert self.compact is not None
        return self.compact
    
    def get_hash(self):
        return self.get_compact()

    def set_compact(self, compact):
        self.compact = compact

    def sample_random_architecture(self, dataset_api):
        raise NotImplementedError()

    def get_type(self):
        return 'nlp'
