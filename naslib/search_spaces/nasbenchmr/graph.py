import numpy as np
import random
import torch
import time

from naslib.search_spaces.core.graph import Graph, EdgeData
from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.nasbenchmr.conversions import (
    convert_params_to_op_indices,
    convert_op_idices_to_params,
    convert_naslib_to_op_indices,
)




# this is a hack currently used to run statistics. 
FIXED_DATA = None
# 'video', '3ddet', 'seg', 'cls-50-1000'

class NasBenchMRSearchSpace(Graph):
    """
    Contains the interface to the tabular benchmark of MR
    """

    QUERYABLE = True
    

    def __init__(self):
        super().__init__()
        self.op_indices = [] # not sure about this line
        self.load_labeled = None
        self.index = None
        self.space_name = "nasbench_MR"

    def load_labeled_architecture(self, dataset_api=None, dataset=FIXED_DATA):
        """
        Currently this method needs access to the dataset.
        Best thing would be to remove that dependence.
        """
        index = np.random.choice(len(dataset_api[dataset]))
        embedding = dataset_api[dataset][index]["embedding"]
        op_indices = convert_params_to_op_indices(embedding=embedding)
        self.set_op_indices(op_indices)
        self.load_labeled = True
        self.set_index(index)
        
    def set_spec(self, index, dataset_api=None, dataset=FIXED_DATA):
        """
        Currently this method needs access to the dataset.
        Best thing would be to remove that dependence.
        """
        embedding = dataset_api[dataset][index]["embedding"]
        op_indices = convert_params_to_op_indices(embedding=embedding)
        self.set_op_indices(op_indices)
        self.load_labeled = True
        self.set_index(index)
        
    def get_arch_iterator(self, dataset_api=None, dataset=FIXED_DATA):
        """
        Currently this method needs access to the dataset.
        Best thing would be to remove that dependence.
        """
        return range(len(dataset_api[dataset]))

    def query(
        self, metric=None,
        dataset=FIXED_DATA,
        path=None,
        epoch=-1,
        full_lc=False,
        dataset_api=None,
    ):
        """
        Query results from ncp.
        """
        start = time.time()
        if metric != Metric.VAL_ACCURACY:
            return -1

        if self.op_indices is None:
            return -1

        embedding = convert_op_idices_to_params(self.op_indices)
        params = convert_op_idices_to_params(self.op_indices)
        if self.load_labeled:
            """
            If we loaded the architecture from the nasbenchmr training data (using
            load_labeled_architecture()), then self.compact will contain the architecture spec,
            and we can query the final train loss from the training data used to train the surrogates 
            """
            embedding_idx = self.get_index()
            query_result = dataset_api[dataset][embedding_idx]
            if dataset == '3ddet':
                result = query_result['moderate']
            elif 'cls' in dataset or 'video' in dataset:
                result = query_result['top1']
            elif 'seg' in dataset:
                result = query_result['mIoU']
        else:
            query_result = dataset_api["api"].query(task=dataset, data_embedding=embedding)
            print(time.time()-start)
            result = query_result.get('main_metric')
        return result

    def get_op_indices(self):
        if self.op_indices is None:
            self.op_indices = convert_naslib_to_op_indices(self)
        return self.op_indices

    def set_op_indices(self, op_indices):
        # This will update the edges in the naslib object to op_indices
        self.op_indices = op_indices
        # convert_op_indices_to_naslib(self)

    def set_index(self, index):
        self.index = index

    def get_index(self):
        return self.index

    def get_hash(self):
        return tuple(self.get_op_indices())

    def sample_random_architecture(self, dataset_api=None):
        """
        This will sample a random architecture.
        Currently a hack is used to ensure randomness across trials.
        """
        st0 = np.random.get_state()
        np.random.seed(int(time.time()))
        op_indices = np.random.randint(2, size=80)
        self.set_op_indices(op_indices)
        np.random.set_state(st0)

    def mutate(self, parent, dataset_api=None):
        """
        This will mutate one op from the parent op indices, and then
        update the naslib object and op_indices.
        Currently a hack is used to ensure randomness across trials.
        """

        parent_op_indices = parent.get_op_indices()
        op_indices = parent_op_indices

        st0 = np.random.get_state()
        np.random.seed(int(time.time()))
        edge = np.random.choice(len(parent_op_indices))
        # flips the current op on the edge
        op_indices[edge] = 1-op_indices[edge]
        self.set_op_indices(op_indices)
        np.random.set_state(st0)

    def find_nbr_params(self, param):
        if param <= 4:
            if param == 1:
                return [2]
            elif param == 4:
                return [3]
            else:
                return [param-1, param+1]
        elif param <= 128:
            if param == 128:
                return [120]
            elif param == 8:
                return [16]
            else:
                return [param-8, param+8]
        else:
            return []

    def get_nbhd(self, dataset_api=None):
        # returns all neighbors of the architecture
        #Currently a hack is used to ensure randomness across trials.
        # arch_embedding = convert_naslib_graph_to_params(self)
        st0 = np.random.get_state()
        np.random.seed(int(time.time()))
        nbr_embeddings = []
        arch_embedding = convert_op_idices_to_params(self.op_indices)
        for param_idx in range(len(arch_embedding)):
            available_params = self.find_nbr_params(arch_embedding[param_idx])
            for nbr_param in available_params:
                nbr_arch_embedding = arch_embedding.copy()
                nbr_arch_embedding[param_idx] = nbr_param
                nbr_op_indices = convert_params_to_op_indices(nbr_arch_embedding)
                nbr_embedding = NasBenchMRSearchSpace()
                nbr_embedding.set_op_indices(nbr_op_indices)
                nbr_model = torch.nn.Module()
                nbr_model.arch = nbr_embedding
                nbr_embeddings.append(nbr_model)


        random.shuffle(nbr_embeddings)
        np.random.set_state(st0)
        return nbr_embeddings

    def get_type(self):
        return "nasbench_MR"