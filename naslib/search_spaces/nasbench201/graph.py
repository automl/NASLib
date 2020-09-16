import os
import pickle
import torch.nn as nn

from naslib.search_spaces.core import primitives as ops
from naslib.search_spaces.core.graph import Graph, EdgeData
from naslib.search_spaces.core.primitives import AbstractPrimitive

from .primitives import ResNetBasicblock, ReLUConvBN

def _set_cell_ops(current_edge_data, C):
    current_edge_data.set('op', [
        ops.Identity(),
        ops.Zero(stride=1),
        ReLUConvBN(C, C, kernel_size=3),
        ReLUConvBN(C, C, kernel_size=1),
        ops.AvgPool1x1(kernel_size=3, stride=1),
    ])
    return current_edge_data


def _truncate_input_edges(node, in_edges, out_edges):
    """
    Discretize the one-shot model.
    """
    if any(e.has('alpha') or (e.has('final') and e.final) for _, e in in_edges):
        # We are in the one-shot case
        for _, data in in_edges:
            if data.has('final') and data.final:
                return  # We are looking at an out node
            data.alpha[1] = -float("Inf")   # Zero op should never be max alpha
        sorted_edge_ids = sorted(in_edges, key=lambda x: max(x[1].alpha), reverse=True)
        keep_edges, _ = zip(*sorted_edge_ids[:])
        for edge_id, edge_data in in_edges:
            if edge_id not in keep_edges:
                edge_data.delete()
    else:
        # We are in the discrete case (e.g. random search)
        k = 2
        for _, data in in_edges:
            assert isinstance(data.op, list)
            data.op.pop(1)      # Remove the zero op
        if any(e.has('final') and e.final for _, e in in_edges):
            return  # TODO: how about mixed final and non-final?
        else:
            for _ in range(len(in_edges) - k): #TODO: this is not correct. Fix it later
                in_edges[random.randint(0, len(in_edges)-1)][1].delete()


class NasBench201SeachSpace(Graph):
    """
    A simplified version of the DARTS cell search space for playing around.
    """

    OPTIMIZER_SCOPE = [
        "stage_1",
        "stage_2",
        "stage_3",
    ]

    def __init__(self):
        super().__init__()

        #
        # Cell definition
        #
        cell = Graph()
        cell.name = "cell"    # Use the same name for all cells with shared attributes

        # Input node
        cell.add_node(1)

        # Intermediate nodes
        cell.add_node(2)
        cell.add_node(3)

        # Output node
        cell.add_node(4)

        # Edges
        cell.add_edges_from([(i, i+1) for i in range(1, 4)])
        cell.add_edges_from([(i, i+2) for i in range(1, 3)])
        cell.add_edges_from([(1, 4)])

        #
        # Makrograph definition
        #
        self.name = "makrograph"

        # Cell is on the edges
        # 1-2:               Preprocessing
        # 2-3, ..., 6-7:     cells stage 1
        # 7-8:               residual block stride 2
        # 8-9, ..., 12-13:   cells stage 2
        # 13-14:             residual block stride 2
        # 14-15, ..., 18-19: cells stage 3
        # 19-20:             post-processing

        total_num_nodes = 20
        self.add_nodes_from(range(1, total_num_nodes+1))
        self.add_edges_from([(i, i+1) for i in range(1, total_num_nodes)])

        channels = [16, 32, 64]

        #
        # operations at the edges
        #

        # preprocessing
        self.edges[1, 2].set('op', ops.Stem(channels[0]))
        
        # stage 1
        for i in range(2, 7):
            self.edges[i, i+1].set('op', cell.copy().set_scope('stage_1'))
        
        # stage 2
        self.edges[7, 8].set('op', ResNetBasicblock(C_in=channels[0], C_out=channels[1], stride=2))
        for i in range(8, 13):
            self.edges[i, i+1].set('op', cell.copy().set_scope('stage_2'))

        # stage 3
        self.edges[13, 14].set('op', ResNetBasicblock(C_in=channels[1], C_out=channels[2], stride=2))
        for i in range(14, 19):
            self.edges[i, i+1].set('op', cell.copy().set_scope('stage_3'))

        # post-processing
        self.edges[19, 20].set('op', ops.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], 10)
        ))
        
        # set the ops at the cells (channel dependent)
        for c, scope in zip(channels, self.OPTIMIZER_SCOPE):
            self.update_edges(
                update_func=lambda current_edge_data: _set_cell_ops(current_edge_data, C=c),
                scope=scope,
                private_edge_data=True
            )



    def prepare_discretization(self):
        self.update_nodes(_truncate_input_edges, scope=self.OPTIMIZER_SCOPE, single_instances=True)
        self.QUERYABLE = True


    def query(self, metric='test_acc', dataset='cifar10', path='../../data'):
        """
            Return e.g.: '|avg_pool_3x3~0|+|nor_conv_1x1~0|skip_connect~1|+|nor_conv_1x1~0|skip_connect~1|skip_connect~2|'
        """
        assert self.QUERYABLE == True
        ops_to_nb201 = {
            'AvgPool1x1': 'avg_pool_3x3',
            'ReLUConvBN1x1': 'nor_conv_1x1',
            'ReLUConvBN3x3': 'nor_conv_3x3',
            'Identity': 'skip_connect',
            'Zero': 'none',
        }

        # convert the naslib representation to nasbench201
        cell = self._get_child_graphs(single_instances=True)[0]
        edge_op_dict = {
            (i, j): ops_to_nb201[cell.edges[i, j]['op'].get_op_name] for i, j in cell.edges
        }
        op_edge_list = [
            '{}~{}'.format(edge_op_dict[(i, j)], i-1) for i, j in sorted(edge_op_dict, key=lambda x: x[1])
        ]

        arch_str = '|{}|+|{}|{}|+|{}|{}|{}|'.format(*op_edge_list)

        # load the nasbench201 data and return the queried data
        with open(os.path.join(path, 'nb201_all.pickle'), 'rb') as f:
            nb201_data = pickle.load(f)
        query_results = nb201_data[arch_str]
        if metric == 'all':
            return query_results[dataset]
        else:
            return query_results[dataset][metric]



    # def query_architecture(self, arch_weights):
    #     arch_weight_idx_to_parent = {0: 0, 1: 0, 2: 1, 3: 0, 4: 1, 5: 2}
    #     arch_strs = {
    #         'cell_normal_from_0_to_1': '',
    #         'cell_normal_from_0_to_2': '',
    #         'cell_normal_from_1_to_2': '',
    #         'cell_normal_from_0_to_3': '',
    #         'cell_normal_from_1_to_3': '',
    #         'cell_normal_from_2_to_3': '',
    #     }
    #     for arch_weight_idx, (edge_key, edge_weights) in enumerate(arch_weights.items()):
    #         edge_weights_norm = torch.softmax(edge_weights, dim=-1)
    #         selected_op_str = PRIMITIVES[edge_weights_norm.argmax()]
    #         arch_strs[edge_key] = '{}~{}'.format(selected_op_str, arch_weight_idx_to_parent[arch_weight_idx])

    #     arch_str = '|{}|+|{}|{}|+|{}|{}|{}|'.format(*arch_strs.values())
    #     if not hasattr(self, 'nasbench_api'):
    #         self.nasbench_api = API('/home/siemsj/nasbench_201.pth')
    #     index = self.nasbench_api.query_index_by_arch(arch_str)
    #     self.nasbench_api.show(index)
    #     info = self.nasbench_api.query_by_index(index)
    #     return self.export_nasbench_201_results_to_dict(info)
