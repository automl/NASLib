import numpy as np
import random
import itertools
import torch
import torch.nn as nn

from naslib.search_spaces.core import primitives as ops
from naslib.search_spaces.nasbench301.primitives import FactorizedReduce
from naslib.search_spaces.core.graph import Graph
from naslib.search_spaces.core.primitives import Sequential
from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.transbench101.conversions import (
    convert_op_indices_to_naslib,
    convert_naslib_to_op_indices,
    convert_naslib_to_transbench101_micro,
    convert_naslib_to_transbench101_macro
)

OP_NAMES = ['Identity', 'Zero', 'ReLUConvBN3x3', 'ReLUConvBN1x1']


class TransBench101SearchSpaceMicro(Graph):
    """
    Implementation of the transbench 101 search space.
    It also has an interface to the tabular benchmark of transbench 101.
    """

    OPTIMIZER_SCOPE = [
        "r_stage_1",
        "n_stage_1",
        "r_stage_2",
        "n_stage_2",
        "r_stage_3"
    ]

    QUERYABLE = True

    def __init__(self, dataset='jigsaw', use_small_model=True,
                 create_graph=False, n_classes=10, in_channels=3):
        super().__init__()
        if dataset == "jigsaw":
            self.num_classes = 1000
        elif dataset == "class_object":
            self.num_classes = 100
        elif dataset == "class_scene":
            self.num_classes = 63
        else:
            self.num_classes = n_classes
        self.op_indices = None

        self.use_small_model = use_small_model
        self.max_epoch = 199
        self.in_channels = in_channels
        self.space_name = 'transbench101'
        self.dataset = dataset

        self.create_graph = create_graph
        self.labeled_archs = None
        self.instantiate_model = True
        self.sample_without_replacement = False

        self.add_edge(1, 2)  # TODO Add condition to call a function to create super graph if create_graph == True

        return
        #
        # Cell definition
        #
        cell = Graph()
        cell.name = "cell"  # Use the same name for all cells with shared attributes

        # Input node
        cell.add_node(1)

        # Intermediate nodes
        cell.add_node(2)
        cell.add_node(3)

        # Output node
        cell.add_node(4)

        # Edges
        cell.add_edges_densly()

        #
        # Makrograph definition
        #
        self.name = "makrograph"

        self.n_modules = 5  # short: 3
        self.blocks_per_module = [2] * self.n_modules  # Change to customize number of blocks per module
        self.module_stages = ["r_stage_1", "n_stage_1", "r_stage_2", "n_stage_2", "r_stage_3"]
        self.base_channels = 64  # short: 16

        n_nodes = 1 + self.n_modules + 1  # Stem, modules, decoder

        # Add nodes and edges
        self.add_nodes_from(range(1, n_nodes + 1))
        for node in range(1, n_nodes):
            self.add_edge(node, node + 1)

        # Preprocessing for jigsaw
        self.edges[1, 2].set('op', self._get_stem_for_task(self.dataset))

        # Add modules
        for idx, node in enumerate(range(2, 2 + self.n_modules)):
            # Create module
            module = self._create_module(self.blocks_per_module[idx], self.module_stages[idx], cell)
            module.set_scope(f"module_{idx + 1}", recursively=False)

            # Add module as subgraph
            self.nodes[node]["subgraph"] = module
            module.set_input([node - 1])

        # Assign operations to cell edges
        C_in = self.base_channels
        for module_node, stage in zip(range(2, 2 + self.n_modules), self.module_stages):
            module = self.nodes[module_node]["subgraph"]
            self._set_cell_ops_for_module(module, C_in, stage)
            C_in = self._get_module_n_output_channels(module)

        # Add decoder depending on the task
        self.edges[node, node + 1].set('op',
                                       self._get_decoder_for_task(self.dataset,
                                                                  n_channels=self._get_module_n_output_channels(module))
                                       )

    def _get_stem_for_task(self, task):
        if task == "jigsaw":
            return ops.StemJigsaw(C_out=self.base_channels)
        elif task in ["class_object", "class_scene"]:
            return ops.Stem(C_out=self.base_channels)
        elif task == "autoencoder":
            return ops.Stem(C_out=self.base_channels)
        else:
            return ops.Stem(C_in=self.in_channels, C_out=self.base_channels)

    def _get_decoder_for_task(self, task, n_channels):  # TODO: Remove harcoding
        if task == "jigsaw":
            return ops.SequentialJigsaw(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(n_channels * 9, self.num_classes)
            )
        elif task in ["class_object", "class_scene"]:
            return ops.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(n_channels, self.num_classes)
            )
        elif task == "autoencoder":
            # return ops.GenerativeDecoder((64, 32), (256, 2048)) # Short
            return ops.GenerativeDecoder((512, 32), (512, 2048))  # Full TNB

        else:
            return None  # TODO: handle other tasks

    def _get_module_n_output_channels(self, module):
        last_cell_in_module = module.edges[1, 2]['op'].op[-1]
        edge_to_last_node = last_cell_in_module.edges[3, 4]
        relu_conv_bn = [op for op in edge_to_last_node['op'] if isinstance(op, ops.ReLUConvBN)][0]
        conv = [m for m in relu_conv_bn.op if isinstance(m, nn.Conv2d)][0]

        return conv.out_channels

    def _is_reduction_stage(self, stage):
        return "r_stage" in stage

    def _set_cell_ops_for_module(self, module, C_in, stage):
        assert isinstance(module, Graph)
        assert module.name == 'module'

        cells = module.edges[1, 2]['op'].op

        for idx, cell in enumerate(cells):
            downsample = self._is_reduction_stage(stage) and idx == 0
            cell.update_edges(
                update_func=lambda edge: _set_op(edge, C_in, downsample),
                private_edge_data=True
            )

            if downsample:
                C_in *= 2

    def _create_module(self, n_blocks, scope, cell):
        blocks = []
        for _ in range(n_blocks):
            blocks.append(cell.copy().set_scope(scope))

        return self._wrap_with_graph(Sequential(*blocks))

    def _wrap_with_graph(self, module):
        container = Graph()
        container.name = 'module'
        container.add_nodes_from([1, 2])
        container.add_edge(1, 2)
        container.edges[1, 2].set('op', module)
        return container

    def query(self, metric=None, dataset=None, path=None, epoch=-1, full_lc=False, dataset_api=None):
        """
        Query results from transbench 101
        """
        assert isinstance(metric, Metric)
        if metric == Metric.ALL:
            raise NotImplementedError()
        if dataset_api is None:
            raise NotImplementedError('Must pass in dataset_api to query transbench101')

        arch_str = convert_naslib_to_transbench101_micro(self)

        query_results = dataset_api['api']
        task = dataset_api['task']

        if task in ['class_scene', 'class_object', 'jigsaw']:

            metric_to_tb101 = {
                Metric.TRAIN_ACCURACY: 'train_top1',
                Metric.VAL_ACCURACY: 'valid_top1',
                Metric.TEST_ACCURACY: 'test_top1',
                Metric.TRAIN_LOSS: 'train_loss',
                Metric.VAL_LOSS: 'valid_loss',
                Metric.TEST_LOSS: 'test_loss',
                Metric.TRAIN_TIME: 'time_elapsed',
            }

        elif task == 'room_layout':

            metric_to_tb101 = {
                Metric.TRAIN_ACCURACY: 'train_neg_loss',
                Metric.VAL_ACCURACY: 'valid_neg_loss',
                Metric.TEST_ACCURACY: 'test_neg_loss',
                Metric.TRAIN_LOSS: 'train_loss',
                Metric.VAL_LOSS: 'valid_loss',
                Metric.TEST_LOSS: 'test_loss',
                Metric.TRAIN_TIME: 'time_elapsed',
            }

        elif task == 'segmentsemantic':

            metric_to_tb101 = {
                Metric.TRAIN_ACCURACY: 'train_acc',
                Metric.VAL_ACCURACY: 'valid_acc',
                Metric.TEST_ACCURACY: 'test_acc',
                Metric.TRAIN_LOSS: 'train_loss',
                Metric.VAL_LOSS: 'valid_loss',
                Metric.TEST_LOSS: 'test_loss',
                Metric.TRAIN_TIME: 'time_elapsed',
            }

        else:  # ['normal', 'autoencoder']

            metric_to_tb101 = {
                Metric.TRAIN_ACCURACY: 'train_ssim',
                Metric.VAL_ACCURACY: 'valid_ssim',
                Metric.TEST_ACCURACY: 'test_ssim',
                Metric.TRAIN_LOSS: 'train_l1_loss',
                Metric.VAL_LOSS: 'valid_l1_loss',
                Metric.TEST_LOSS: 'test_l1_loss',
                Metric.TRAIN_TIME: 'time_elapsed',
            }

        if metric == Metric.RAW:
            # return all data
            return query_results.get_arch_result(arch_str).query_all_results()[task]

        if metric == Metric.HP:
            # return hyperparameter info
            return query_results[dataset]['cost_info']
        elif metric == Metric.TRAIN_TIME:
            return query_results.get_single_metric(arch_str, task, metric_to_tb101[metric])

        if full_lc and epoch == -1:
            return query_results[dataset][metric_to_tb101[metric]]
        elif full_lc and epoch != -1:
            return query_results[dataset][metric_to_tb101[metric]][:epoch]
        else:
            return query_results.get_single_metric(arch_str, task, metric_to_tb101[metric])

    def get_op_indices(self):
        if self.op_indices is None:
            self.op_indices = convert_naslib_to_op_indices(self)
        return self.op_indices

    def get_hash(self):
        return tuple(self.get_op_indices())

    def set_op_indices(self, op_indices):
        # This will update the edges in the naslib object to op_indices
        self.op_indices = op_indices
        convert_op_indices_to_naslib(op_indices, self)

    def get_arch_iterator(self, dataset_api=None):
        return itertools.product(range(4), repeat=6)

    def set_spec(self, op_indices, dataset_api=None):
        # this is just to unify the setters across search spaces
        # TODO: change it to set_spec on all search spaces
        self.set_op_indices(op_indices)

    def sample_random_architecture(self, dataset_api=None):
        """
        This will sample a random architecture and update the edges in the
        naslib object accordingly.
        """
        op_indices = np.random.randint(4, size=(6))
        self.set_op_indices(op_indices)

    def mutate(self, parent, dataset_api=None):
        """
        This will mutate one op from the parent op indices, and then
        update the naslib object and op_indices
        """
        parent_op_indices = parent.get_op_indices()
        op_indices = list(parent_op_indices)

        edge = np.random.choice(len(parent_op_indices))
        available = [o for o in range(len(OP_NAMES)) if o != parent_op_indices[edge]]
        op_index = np.random.choice(available)
        op_indices[edge] = op_index
        print('op_indices mu =', op_indices)
        self.set_op_indices(op_indices)

    def get_nbhd(self, dataset_api=None):
        # return all neighbors of the architecture
        self.get_op_indices()
        nbrs = []
        for edge in range(len(self.op_indices)):
            available = [o for o in range(len(OP_NAMES)) if o != self.op_indices[edge]]

            for op_index in available:
                nbr_op_indices = list(self.op_indices).copy()
                nbr_op_indices[edge] = op_index
                nbr = TransBench101SearchSpaceMicro()
                nbr.set_op_indices(nbr_op_indices)
                nbr_model = torch.nn.Module()
                nbr_model.arch = nbr
                nbrs.append(nbr_model)

        random.shuffle(nbrs)
        return nbrs

    def get_type(self):
        return 'transbench101'


class TransBench101SearchSpaceMacro(Graph):
    """
    Implementation of the transbench 101 search space.
    It also has an interface to the tabular benchmark of transbench 101.
    """

    OPTIMIZER_SCOPE = [
        "stage_1",
        "stage_2",
        "stage_3",
    ]

    QUERYABLE = True

    def __init__(self, dataset='jigsaw', *arg, **kwargs):
        super().__init__()
        if dataset == "jigsaw":
            self.num_classes = 1000
        elif dataset == "class_object":
            self.num_classes = 100
        elif dataset == "class_scene":
            self.num_classes = 63
        else:
            self.num_classes = -1

        self.dataset = dataset
        self.op_indices = None

        self.max_epoch = 199
        self.space_name = 'transbench101'
        self.labeled_archs = None
        self.instantiate_model = True
        self.sample_without_replacement = False

        self.add_edge(1, 2)

    def query(self, metric=None, dataset=None, path=None, epoch=-1, full_lc=False, dataset_api=None):
        """
        Query results from transbench 101
        """
        assert isinstance(metric, Metric)
        if metric == Metric.ALL:
            raise NotImplementedError()
        if dataset_api is None:
            raise NotImplementedError('Must pass in dataset_api to query transbench101')

        arch_str = convert_naslib_to_transbench101_macro(self.op_indices)

        query_results = dataset_api['api']
        task = dataset_api['task']

        if task in ['class_scene', 'class_object', 'jigsaw']:

            metric_to_tb101 = {
                Metric.TRAIN_ACCURACY: 'train_top1',
                Metric.VAL_ACCURACY: 'valid_top1',
                Metric.TEST_ACCURACY: 'test_top1',
                Metric.TRAIN_LOSS: 'train_loss',
                Metric.VAL_LOSS: 'valid_loss',
                Metric.TEST_LOSS: 'test_loss',
                Metric.TRAIN_TIME: 'time_elapsed',
            }

        elif task == 'room_layout':

            metric_to_tb101 = {
                Metric.TRAIN_ACCURACY: 'train_neg_loss',
                Metric.VAL_ACCURACY: 'valid_neg_loss',
                Metric.TEST_ACCURACY: 'test_neg_loss',
                Metric.TRAIN_LOSS: 'train_loss',
                Metric.VAL_LOSS: 'valid_loss',
                Metric.TEST_LOSS: 'test_loss',
                Metric.TRAIN_TIME: 'time_elapsed',
            }

        elif task == 'segmentsemantic':

            metric_to_tb101 = {
                Metric.TRAIN_ACCURACY: 'train_acc',
                Metric.VAL_ACCURACY: 'valid_acc',
                Metric.TEST_ACCURACY: 'test_acc',
                Metric.TRAIN_LOSS: 'train_loss',
                Metric.VAL_LOSS: 'valid_loss',
                Metric.TEST_LOSS: 'test_loss',
                Metric.TRAIN_TIME: 'time_elapsed',
            }

        else:  # ['normal', 'autoencoder']

            metric_to_tb101 = {
                Metric.TRAIN_ACCURACY: 'train_ssim',
                Metric.VAL_ACCURACY: 'valid_ssim',
                Metric.TEST_ACCURACY: 'test_ssim',
                Metric.TRAIN_LOSS: 'train_loss',
                Metric.VAL_LOSS: 'valid_loss',
                Metric.TEST_LOSS: 'test_loss',
                Metric.TRAIN_TIME: 'time_elapsed',
            }

        if metric == Metric.RAW:
            # return all data
            return query_results.get_arch_result(arch_str).query_all_results()[task]

        if metric == Metric.HP:
            # return hyperparameter info
            return query_results[dataset]['cost_info']
        elif metric == Metric.TRAIN_TIME:
            return query_results.get_single_metric(arch_str, task, metric_to_tb101[metric])

        if full_lc and epoch == -1:
            return query_results[dataset][metric_to_tb101[metric]]
        elif full_lc and epoch != -1:
            return query_results[dataset][metric_to_tb101[metric]][:epoch]
        else:
            return query_results.get_single_metric(arch_str, task, metric_to_tb101[metric])

    def get_op_indices(self):
        if self.op_indices is None:
            self.op_indices = convert_naslib_to_op_indices(self)
        return self.op_indices

    def get_hash(self):
        return tuple(self.get_op_indices())

    def set_op_indices(self, op_indices):
        # This will update the edges in the naslib object to op_indices
        self.op_indices = op_indices

    def sample_random_architecture(self, dataset_api=None):
        """
        This will sample a random architecture and update the edges in the
        naslib object accordingly.
        """
        r = random.randint(0, 2)
        p = random.randint(1, 4)
        q = random.randint(1, 3)
        u = [2 * int(i < p) for i in range(r + 4)]
        v = [int(i < q) for i in range(r + 4)]
        w = [1 + sum(x) for x in zip(u, v)]
        op_indices = np.random.permutation(w)
        while len(op_indices) < 6:
            op_indices = np.append(op_indices, 0)
        self.set_op_indices(op_indices)

    def mutate(self, parent, dataset_api=None):
        """
        This will mutate one op from the parent op indices, and then
        update the naslib object and op_indices
        """
        parent_op_indices = list(parent.get_op_indices())
        parent_op_ind = parent_op_indices[parent_op_indices != 0]

        def f(g):
            r = len(g)
            p = sum([int(i == 4 or i == 3) for i in g])
            q = sum([int(i == 4 or i == 2) for i in g])
            return r, p, q

        def g(r, p, q):
            u = [2 * int(i < p) for i in range(r)]
            v = [int(i < q) for i in range(r)]
            w = [1 + sum(x) for x in zip(u, v)]
            return np.random.permutation(w)

        a, b, c = f(parent_op_ind)

        a_available = [i for i in [4, 5, 6] if i != a]
        b_available = [i for i in range(1, 5) if i != b]
        c_available = [i for i in range(1, 4) if i != c]

        dic1 = {1: a, 2: b, 3: c}
        dic2 = {1: a_available, 2: b_available, 3: c_available}

        numb = random.randint(1, 3)

        dic1[numb] = random.choice(dic2[numb])

        op_indices = g(dic1[1], dic1[2], dic1[3])
        while len(op_indices) < 6:
            op_indices = np.append(op_indices, 0)

        self.set_op_indices(op_indices)

    def get_nbhd(self, dataset_api=None):
        # return all neighbors of the architecture
        self.get_op_indices()
        op_ind = list(self.op_indices[self.op_indices != 0])
        nbrs = []

        def f(g):
            r = len(g)
            p = sum([int(i == 4 or i == 3) for i in g])
            q = sum([int(i == 4 or i == 2) for i in g])
            return r, p, q

        def g(r, p, q):
            u = [2 * int(i < p) for i in range(r)]
            v = [int(i < q) for i in range(r)]
            w = [1 + sum(x) for x in zip(u, v)]
            return np.random.permutation(w)

        a, b, c = f(op_ind)

        a_available = [i for i in [4, 5, 6] if i != a]
        b_available = [i for i in range(1, 5) if i != b]
        c_available = [i for i in range(1, 4) if i != c]

        for r in a_available:
            for p in b_available:
                for q in c_available:
                    nbr_op_indices = g(r, p, q)
                    while len(nbr_op_indices) < 6:
                        nbr_op_indices = np.append(nbr_op_indices, 0)
                    nbr = TransBench101SearchSpaceMacro()
                    nbr.set_op_indices(nbr_op_indices)
                    nbr_model = torch.nn.Module()
                    nbr_model.arch = nbr
                    nbrs.append(nbr_model)

        random.shuffle(nbrs)
        return nbrs

    def get_type(self):
        return 'transbench101'


def _set_op(edge, C_in, downsample):
    C_out = C_in
    stride = 1

    if downsample:
        if edge.head == 1:
            C_out = C_in * 2
            stride = 2
        else:
            C_in *= 2
            C_out = C_in
            stride = 1

    edge.data.set("op", [
        ops.Identity() if stride == 1 else FactorizedReduce(C_in, C_out, stride, affine=False),
        ops.Zero(stride=stride, C_in=C_in, C_out=C_out),
        ops.ReLUConvBN(C_in, C_out, kernel_size=3, stride=stride),
        ops.ReLUConvBN(C_in, C_out, kernel_size=1, stride=stride),
    ])
