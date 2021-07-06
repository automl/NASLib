import os
import pickle
import numpy as np
import copy
import random

from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.core.graph import Graph
from naslib.utils.utils import get_project_root


class NasBenchASRSearchSpace(Graph):
    """
    Contains the interface to the tabular benchmark of nas-bench-asr.
    Note: currently we do not support building a naslib object for
    nas-bench-asr architectures.
    """

    QUERYABLE = False

    def __init__(self):
        super().__init__()

    def load_labeled_architecture(self, dataset_api=None, max_nodes=25):
        """
        This is meant to be called by a new NasBenchASRSearchSpace() object.
        It samples a random architecture from the nas-bench-asr data.
        """
        pass

    def query(self, metric=None, dataset=None, path=None, epoch=-1, full_lc=False, dataset_api=None):
        """
        Query results from nas-bench-asr
        """
        pass

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
        return 'asr'
