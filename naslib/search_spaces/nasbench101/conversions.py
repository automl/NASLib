from naslib.utils.utils import AttrDict
from typing import List, Tuple

import copy
import numpy as np
import torch.nn.functional as F
from naslib.search_spaces.core import primitives as ops
from naslib.search_spaces.core.graph import Graph

"""
'naslib': the NASBench101SearchSpace object
'spec': adjacency matrix + op list
"""


INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NODES_IN_DEGREE = [1, 1, 1, 2, 2, 2] # Search space 3 as defined in NAS-Bench-1Shot1 (https://arxiv.org/abs/2001.10422)


def _truncate_input_edges(node: Tuple[int, object], in_edges: List[object], out_edges: List[object]) -> None:
    """
    Removes input edges if there are more than the number of permissible edges as defined in NODES_IN_DEGREE.

    Args:
        node        : Tuple of (node_id, node_data)
        in_edges    : List of incoming edges to node
        out_edges   : List of outgoing edges to node

    Returns:
        None
    """
    node_idx, _ = node

    if node_idx %2 != 0: # We're only interested in connections to even (summation) nodes
        return

    def _largest_post_softmax_weight(edge):
        _, edge_data = edge
        alpha_softmax = F.softmax(edge_data.alpha.detach())

        # Return the softmax activation for identity operation only (the first alpha)
        return alpha_softmax[0]


    in_degree_for_node = NODES_IN_DEGREE[node_idx//2 - 1]

    if len(in_edges) > in_degree_for_node:
        if any(e.has("alpha") or (e.has("final") and e.final) for _, e in in_edges):
            sorted_edge_ids = sorted(in_edges, key=_largest_post_softmax_weight, reverse=True)
            keep_edges, _ = zip(*sorted_edge_ids[:in_degree_for_node])
            for edge_id, edge_data in in_edges:
                if edge_id not in keep_edges:
                    edge_data.delete()

def _discretize_ops(edge: AttrDict) -> None:
    """
    Replaces the operation on an edge with the operation with the highest alpha.

    Args:
        edge    : Edge containing the operation
    """
    if edge.data.has("alpha"):
        primitives = edge.data.op.get_embedded_ops()

        # If the MixedOp has already been replaced with the primitive, then
        # there's nothing for us to do here.
        if primitives == None:
            return

        alphas = edge.data.alpha.detach().cpu()
        edge.data.set("op", primitives[np.argmax(alphas)])

def _convert_final_cell_to_spec(naslib_cell: Graph) -> Tuple[np.ndarray, List[str]]:
    """
    Creates the spec of a given NASLib cell graph

    Args:
        naslib_cell : NASLib Graph object representing the cell in the search space
    """
    # Create a zero matrix
    matrix = np.zeros((7, 7), dtype=int)

    for u, v in naslib_cell.edges():
        if v % 2 != 0: # We're only interested in edges going to summation nodes (only they have multiple incoming edges)
            continue

        u_index = u // 2
        v_index = v // 2

        matrix[u_index][v_index] = 1

    operations = [INPUT]
    # Odd nodes have the node_pair graph as subgraph, which has the operation on its only edge
    for node_idx in range(3, 12, 2):
        node_pair = naslib_cell.nodes[node_idx]['subgraph']
        node_op = node_pair.edges[1, 2]['op']

        if isinstance(node_op, ops.ConvBnReLU):
            if node_op.kernel_size == (3, 3) or node_op.kernel_size == 3:
                operations.append(CONV3X3)
            elif node_op.kernel_size == (1, 1) or node_op.kernel_size == 1:
                operations.append(CONV1X1)
            else:
                raise Exception(f"Convolution operation on Graph has unsupported kernel size")

        elif isinstance(node_op, ops.MaxPool) and (node_op.kernel_size == 3 or node_op.kernel_size == 1):
            operations.append(MAXPOOL3X3)
        else:
            raise Exception(f"Cell edge has unsupported operation")

    operations.append(OUTPUT)

    return {'matrix': matrix, 'ops': operations}

def convert_naslib_to_spec(naslib_object) -> Tuple[np.ndarray, List[str]]:
    """
    Converts a NASLib NASBench101 graph object to its spec, with which NASBench101 can be queried.

    Args:
        naslib_object: NasBench101SearchSpace object

    Returns:
        Spec of the given naslib_object
    """
    # Get the cell graph from the stack
    # Assuming the second edge always contains the stack
    stack = naslib_object.get_all_edge_data('op', scope='macro')[1]
    cell = copy.deepcopy(stack.op[0])

    assert cell.name == 'cell', 'NASBench101 cell not acquired!'

    # Discretize edges of cell, i.e., {Identity, Zero} edges using NAS-Bench-1Shot1 rules
    cell.update_nodes(_truncate_input_edges, scope="cell", single_instances=True)

    # Discretize edges of the node-pair graphs, i.e., {conv3x3-bn-relu, conv1x1-bn-relu, maxpool3x3}
    cell.update_edges(_discretize_ops, scope="node_pair", private_edge_data=True)

    # Convert the cell to the spec
    return _convert_final_cell_to_spec(cell)


def convert_spec_to_naslib(spec, naslib_object):
    # TODO: write this method similar to how it was written for nasbench201 and darts
    raise NotImplementedError("Cannot yet convert a spec to naslib object")


def convert_spec_to_tuple(spec):
    # convert the spec to a hashable type
    op_dict = ["input", "output", "maxpool3x3", "conv1x1-bn-relu", "conv3x3-bn-relu"]

    matrix = spec["matrix"].flatten()
    ops = [op_dict.index(s) for s in spec["ops"]]
    tup = tuple([*matrix, *ops])
    return tup
