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
