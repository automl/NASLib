import torch.nn as nn
from naslib.search_spaces.core.graph import Graph
from naslib.search_spaces.nasbench101.conversions import get_children
from naslib.search_spaces.nasbench101.primitives import ModelWrapper
from naslib.search_spaces.transbench101.tnb101.model_builder import create_model

"""
There are three representations
'naslib': the NASBench201SearchSpace object
'op_indices': A list of six ints, which is the simplest representation
'arch_str': The string representation used in the original nasbench201 paper
This file currently has the following conversions:
naslib -> op_indices
op_indices -> naslib
naslib -> arch_str
Note: we could add more conversions, but this is all we need for now
"""

OP_NAMES = ['Identity', 'Zero', 'ReLUConvBN3x3', 'ReLUConvBN1x1']
EDGE_LIST = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))


def convert_naslib_to_op_indices(naslib_object):
    cell = naslib_object._get_child_graphs(single_instances=True)[0]
    ops = []
    for i, j in EDGE_LIST:
        ops.append(cell.edges[i, j]['op'].get_op_name)

    return [OP_NAMES.index(name) for name in ops]


def _wrap_model(model):
    all_leaf_modules = get_children(model)
    inplace_relus = [module for module in all_leaf_modules if (isinstance(module, nn.ReLU) and module.inplace == True)]

    for relu in inplace_relus:
        relu.inplace = False

    model_wrapper = ModelWrapper(model)

    return model_wrapper


def convert_op_indices_macro_to_model(op_indices, task):
    arch_str = convert_op_indices_macro_to_str(op_indices)
    model = create_model(arch_str, task)

    return _wrap_model(model)


def convert_op_indices_micro_to_model(op_indices, task):
    arch_str = convert_op_indices_micro_to_str(op_indices)
    model = create_model(arch_str, task)

    return _wrap_model(model)


def convert_op_indices_to_naslib(op_indices, naslib_object):
    """
    Converts op indices to a naslib object
    input: op_indices (list of six ints)
    naslib_object is an empty NasBench201SearchSpace() object.
    Do not call this method with a naslib object that has already been 
    discretized (i.e., all edges have a single op).
    output: none, but the naslib object now has all edges set
    as in genotype.
    
    warning: this method will modify the edges in naslib_object.
    """

    # create a dictionary of edges to ops
    edge_op_dict = {}
    for i, index in enumerate(op_indices):
        edge_op_dict[EDGE_LIST[i]] = OP_NAMES[index]

    def add_op_index(edge):
        # function that adds the op index from the dictionary to each edge
        if (edge.head, edge.tail) in edge_op_dict:
            for i, op in enumerate(edge.data.op):
                if op.get_op_name == edge_op_dict[(edge.head, edge.tail)] or (
                        op.get_op_name == 'FactorizedReduce' and edge_op_dict[(edge.head, edge.tail)] == 'Identity'):
                    index = i
                    break
            edge.data.set('op_index', index, shared=True)

    def update_ops(edge):
        # function that replaces the primitive ops at the edges with the one in op_index
        if isinstance(edge.data.op, list):
            primitives = edge.data.op
        else:
            primitives = edge.data.primitives

        edge.data.set('op', primitives[edge.data.op_index])
        edge.data.set('primitives', primitives)  # store for later use

    naslib_object.update_edges(
        add_op_index,
        scope=naslib_object.OPTIMIZER_SCOPE,
        private_edge_data=False
    )

    naslib_object.update_edges(
        update_ops,
        scope=naslib_object.OPTIMIZER_SCOPE,
        private_edge_data=True
    )


def convert_naslib_to_str(naslib_object):
    """
    Converts naslib object to string representation.
    """

    ops_to_nb201 = {
        'AvgPool1x1': 'avg_pool_3x3',
        'ReLUConvBN1x1': 'nor_conv_1x1',
        'ReLUConvBN3x3': 'nor_conv_3x3',
        'Identity': 'skip_connect',
        'Zero': 'none',
    }

    cell = naslib_object.nodes[2]['subgraph'].edges[1, 2]['op'].op[1]  # TODO: Do this in a clean fashion
    assert cell.name == "cell" and isinstance(cell, Graph)

    edge_op_dict = {
        (i, j): ops_to_nb201[cell.edges[i, j]['op'].get_op_name] for i, j in cell.edges
    }
    op_edge_list = [
        '{}~{}'.format(edge_op_dict[(i, j)], i - 1) for i, j in sorted(edge_op_dict, key=lambda x: x[1])
    ]

    return '|{}|+|{}|{}|+|{}|{}|{}|'.format(*op_edge_list)


def convert_naslib_to_transbench101_micro(naslib_object):
    """
    Converts naslib object to string representation.
    To be used later used later with one-shot optimizers 
    """

    ops_to_tb101 = {
        'ReLUConvBN1x1': '2',
        'ReLUConvBN3x3': '3',
        'Identity': '1',
        'Zero': '0',
    }

    cell = naslib_object.nodes[2]['subgraph'].edges[1, 2]['op'].op[1]
    assert cell.name == "cell" and isinstance(cell, Graph)

    edge_op_dict = {
        (i, j): ops_to_tb101[cell.edges[i, j]['op'].get_op_name] for i, j in cell.edges
    }

    op_edge_list = [
        '{}'.format(edge_op_dict[(i, j)]) for i, j in sorted(edge_op_dict, key=lambda x: x[1])
    ]

    return '64-41414-{}_{}{}_{}{}{}'.format(*op_edge_list)


# def convert_naslib_to_transbench101_micro(op_indices):
#     """
#     Converts naslib object to string representation.
#     """
#     return '64-41414-{}_{}{}_{}{}{}'.format(*op_indices)


def convert_op_indices_micro_to_str(op_indices):
    """
    Converts naslib object to string representation.
    """
    return '64-41414-{}_{}{}_{}{}{}'.format(*op_indices)


def convert_op_indices_macro_to_str(op_indices):
    """
    Converts naslib object to string representation.
    """
    ops_string = ''.join([str(e) for e in op_indices if e != 0])
    return '64-{}-basic'.format(ops_string)
