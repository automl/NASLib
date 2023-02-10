import os
import pickle
import numpy as np
import copy
import random
import torch

from naslib.search_spaces.core import primitives as core_ops
from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.core.graph import Graph
from naslib.search_spaces.nasbenchasr.primitives import CellLayerNorm, Head, ops, PadConvReluNorm
from naslib.utils import get_project_root
from naslib.search_spaces.nasbenchasr.conversions import flatten, \
    copy_structure, make_compact_mutable, make_compact_immutable
from naslib.search_spaces.nasbenchasr.encodings import encode_asr
from naslib.utils.encodings import EncodingType

OP_NAMES = ['linear', 'conv5', 'conv5d2', 'conv7', 'conv7d2', 'zero']


class NasBenchASRSearchSpace(Graph):
    """
    Contains the interface to the tabular benchmark of nas-bench-asr.
    Note: currently we do not support building a naslib object for
    nas-bench-asr architectures.
    """

    QUERYABLE = True
    OPTIMIZER_SCOPE = [
        'cells_stage_1',
        'cells_stage_2',
        'cells_stage_3',
        'cells_stage_4'
    ]

    def __init__(self):
        super().__init__()
        self.load_labeled = False
        self.max_epoch = 40
        self.max_nodes = 3
        self.accs = None
        self.compact = None

        self.n_blocks = 4
        self.n_cells_per_block = [3, 4, 5, 6]
        self.features = 80
        self.filters = [600, 800, 1000, 1200]
        self.cnn_time_reduction_kernels = [8, 8, 8, 8]
        self.cnn_time_reduction_strides = [1, 1, 2, 2]
        self.scells_per_block = [3, 4, 5, 6]
        self.num_classes = 48
        self.dropout_rate = 0.0
        self.use_norm = True

        self._create_macro_graph()

    def _create_macro_graph(self):
        cell = self._create_cell()

        # Macrograph defintion
        n_nodes = self.n_blocks + 2
        self.add_nodes_from(range(1, n_nodes + 1))

        for node in range(1, n_nodes):
            self.add_edge(node, node + 1)

        # Create the cell blocks and add them as subgraphs of nodes 2 ... 5
        for idx, node in enumerate(range(2, 2 + self.n_blocks)):
            scope = f'cells_stage_{idx + 1}'
            cells_block = self._create_cells_block(cell, n=self.n_cells_per_block[idx], scope=scope)
            self.nodes[node]['subgraph'] = cells_block.set_input([node - 1])

            # Assign the list of operations to the cell edges
            cells_block.update_edges(
                update_func=lambda edge: _set_cell_edge_ops(edge, filters=self.filters[idx], use_norm=self.use_norm),
                scope=scope,
                private_edge_data=True
            )

        # Assign the PadConvReluNorm operation to the edges of the macro graph
        start_node = 1
        for idx, node in enumerate(range(start_node, start_node + self.n_blocks)):
            op = PadConvReluNorm(
                in_channels=self.features if node == start_node else self.filters[idx - 1],
                out_channels=self.filters[idx],
                kernel_size=self.cnn_time_reduction_kernels[idx],
                dilation=1,
                strides=self.cnn_time_reduction_strides[idx],
                groups=1,
                name=f'conv_{idx}'
            )

            self.edges[node, node + 1].set('op', op)

        # Assign the LSTM + Linear layer to the last edge in the macro graph
        head = Head(self.dropout_rate, self.filters[-1], self.num_classes)
        self.edges[self.n_blocks + 1, self.n_blocks + 2].set('op', head)

    def _create_cells_block(self, cell, n, scope):
        block = Graph()
        block.name = f'{n}_cells_block'

        block.add_nodes_from(range(1, n + 2))

        for node in range(2, n + 2):
            block.add_node(node, subgraph=cell.copy().set_scope(scope).set_input([node - 1]))

        for node in range(1, n + 2):
            block.add_edge(node, node + 1)

        return block

    def _create_cell(self):
        cell = Graph()
        cell.name = 'cell'
        # ASR Cell requires two edges between two consecutive nodes, which isn't supported by the NASLib Graph.
        # Solution: use three nodes and their edges to represent two nodes with two edges between them as follows:
        # Desired:
        #   Two edges between Node 1 and Node 2. The first edge to hold {Zero, Id} and the second edge to hold {Linear, 1x5 conv, ... Zero}
        # Solution:
        #   Add nodes 1, 2 and 3.
        #   Edge 1-2 has Id
        #   Edge 2-3 holds {Linear, 1x5 conv, ... Zero}
        #   Edge 1-3 holds {Zero, Id}

        cell.add_nodes_from(range(1, 8))

        # Create edges
        for i in range(1, 7):
            cell.add_edge(i, i + 1)

        for i in range(1, 6, 2):
            for j in range(i + 2, 8, 2):
                cell.add_edge(i, j)

        cell.add_node(8)
        cell.add_edge(7, 8)  # For optional layer normalization

        return cell

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
        query_results = dataset_api["asr_data"].full_info(self.compact)

        if metric != Metric.VAL_ACCURACY:
            if metric == Metric.TEST_ACCURACY:
                return query_results[metric_to_asr[metric]]
            elif (metric == Metric.PARAMETERS) or (metric == Metric.FLOPS):
                return query_results['info'][metric_to_asr[metric]]
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
        self.compact = make_compact_immutable(compact)

    def sample_random_architecture(self, dataset_api):
        search_space = [[len(OP_NAMES)] + [2] * (idx + 1) for idx in
                        range(self.max_nodes)]
        flat = flatten(search_space)
        m = [random.randrange(opts) for opts in flat]
        m = copy_structure(m, search_space)

        compact = m
        self.set_compact(compact)
        return compact

    def mutate(self, parent, mutation_rate=1, dataset_api=None):
        """
        This will mutate the cell in one of two ways:
        change an edge; change an op.
        Todo: mutate by adding/removing nodes.
        Todo: mutate the list of hidden nodes.
        Todo: edges between initial hidden nodes are not mutated.
        """
        parent_compact = parent.get_compact()
        parent_compact = make_compact_mutable(parent_compact)
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

    def get_nbhd(self, dataset_api=None):
        """
        Return all neighbors of the architecture
        """
        compact = self.get_compact()
        # edges, ops, hiddens = compact
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
                        new_compact = make_compact_mutable(new_compact)
                        new_compact[node_id][0] = op_id
                        nbhd = add_to_nbhd(new_compact, nbhd)
                else:
                    edge_op = compact[node_id][edge_id]
                    new_edge_op = int(not edge_op)
                    new_compact = copy.deepcopy(compact)
                    new_compact = make_compact_mutable(new_compact)
                    new_compact[node_id][edge_id] = new_edge_op
                    nbhd = add_to_nbhd(new_compact, nbhd)

        random.shuffle(nbhd)
        return nbhd

    def get_type(self):
        return 'asr'

    def get_max_epochs(self):
        return 39

    def encode(self, encoding_type=EncodingType.ADJACENCY_ONE_HOT):
        return encode_asr(self, encoding_type=encoding_type)


def _set_cell_edge_ops(edge, filters, use_norm):
    if use_norm and edge.head == 7:
        edge.data.set('op', CellLayerNorm(filters))
        edge.data.finalize()
    elif edge.head % 2 == 0:  # Edge from intermediate node
        edge.data.set(
            'op', [
                ops['linear'](filters, filters),
                ops['conv5'](filters, filters),
                ops['conv5d2'](filters, filters),
                ops['conv7'](filters, filters),
                ops['conv7d2'](filters, filters),
                ops['zero'](filters, filters)
            ]
        )
    elif edge.tail % 2 == 0:  # Edge to intermediate node. Should always be Identity.
        edge.data.finalize()
    else:
        edge.data.set(
            'op',
            [
                core_ops.Zero(stride=1),
                core_ops.Identity()
            ]
        )
