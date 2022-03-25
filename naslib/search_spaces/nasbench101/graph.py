import os
import pickle
import numpy as np
import copy
import random
import torch
import torch.nn as nn

from naslib.search_spaces.core import primitives as ops
from naslib.search_spaces.core.graph import Graph, EdgeData
from naslib.search_spaces.core.primitives import ReLUConvBN, MaxPool1x1
from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.nasbench101.conversions import (
    convert_naslib_to_spec,
    convert_spec_to_naslib,
    convert_spec_to_tuple,
)

from typing import Any, List, Tuple, Union, Dict
from naslib.utils.utils import get_project_root

from .primitives import ReLUConvBN

INPUT = "input"
OUTPUT = "output"
CONV3X3 = "conv3x3-bn-relu"
CONV1X1 = "conv1x1-bn-relu"
MAXPOOL3X3 = "maxpool3x3"
OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]

NUM_VERTICES = 7
OP_SPOTS = NUM_VERTICES - 2
MAX_EDGES = 9


class NasBench101SearchSpace(Graph):
    """
    Contains the interface to the tabular benchmark of nasbench 101.
    """

    OPTIMIZER_SCOPE = [
        "node_pair",
        "cell",
    ]

    QUERYABLE = True

    def __init__(self, n_classes=10, stacks:int=3, channels:int=64):
        super().__init__()
        self.num_classes = n_classes

        # Settings for NASBench101 as described in the paper
        self.stacks = stacks
        self.channels = channels
        self.space_name = "nasbench101"
        self.spec = None

        self._create_macro_graph(self.stacks, self.channels, self.num_classes)

    def _create_macro_graph(self, n_stacks:int, in_channels_stack:int, num_classes:int) -> None:
        """
        Creates the macro graph.

        Args:
            n_stacks            : Number of stacks in the macro graph
            in_channels_stack   : Number of input channels to the very first stack (after stem)
            num_classes         : Number of classes of linear output layer

        Returns:
            None. Sets self to the given macro graph.
        """
        self.name = "makrograph"
        self.set_scope("macro")

        # Graph structure
        # 1-2               : stem
        # 2-3, 3-4, 4-5     : stacks (when n_stacks=3)
        # 5-6               : avgpool + dense

        # Create nodes
        n_nodes = 3 + n_stacks
        self.add_nodes_from(range(1, n_nodes+1))

        # Add edges
        for i in range(1, n_nodes):
            self.add_edge(i, i+1)

        # Add edge operations
        self.edges[1, 2].set("op", ops.Stem(C_out=in_channels_stack))

        # Create the cell graph
        # All the cells in the macro graph should be copies of this cell (cell.copy())
        # https://networkx.org/documentation/stable/reference/classes/generated/networkx.Graph.copy.html
        cell = self._create_cell_graph()

        # Add the stacks as edge operations
        stacks_output_node = 2+n_stacks
        in_channels, out_channels = in_channels_stack, in_channels_stack

        for i in range(2, stacks_output_node):
            stack = self._create_stack(in_channels, out_channels, cell)
            in_channels, out_channels = out_channels, out_channels*2
            self.edges[i, i+1].set("op", stack)

        # Add global pooling and dense layer in the final edge
        pool_and_linear = ops.Sequential(
            ops.GlobalAveragePooling(),
            nn.Linear(in_channels, num_classes)
        )

        self.edges[stacks_output_node, stacks_output_node+1].set("op", pool_and_linear)


    def _create_stack(self, in_channels: int, out_channels: int, cell: Graph, downsample=True) -> ops.Sequential:
        """
        Creates a stack with three cells and an optional downsampling operation.

        Args:
            in_channels     : Number of input channels to the cell
            out_channels    : Number of output channels of the cell
            cell            : Graph representation of the cell, copies of which will be used in the stack
            downsample      : Whether or not to use downsampling operation at the end of the stack

        Returns:
            A stack with three cells and, optionally, downsampling operation (maxpool3x3).
        """
        cells = [
            self._create_cell(cell.copy(), in_channels, out_channels),
            self._create_cell(cell.copy(), out_channels, out_channels),
            self._create_cell(cell.copy(), out_channels, out_channels),
        ]

        for cell in cells:
            # Set inputs for the subgraph. This cannot be done sooner because cell.copy() doesn't copy the
            # attributes of the NASLib Graph object (like Graph.input_node_idxs, which is set using Graph.set_input())
            for i in range(3, 13, 2):
                cell.nodes[i]['subgraph'].set_input([i-1])

        if downsample:
            cells.append(nn.MaxPool2d(kernel_size=3))

        return ops.Sequential(*cells)


    def _create_node_pair_graph(self, parent_node:int) -> Graph:
        """
        Creates a node_pair graph, with two nodes and a single edge between them. This edge is used to hold the
        mixed operations as specified in NASBench101 search space (conv3x3, conv1x1, maxpool3x3). This makes it
        possible to represent graphs where nodes hold operations while still using edges to hold them (the
        node_pair graph will simply be a subgraph of the node which represents the operation)

        Args:
            parent_node : Id of parent node

        Returns:
            Node_pair graph.
        """
        node_pair = Graph()
        node_pair.name = "node_pair" + str(parent_node)
        node_pair.set_scope("node_pair")

        node_pair.add_nodes_from([1, 2])
        node_pair.add_edge(1, 2)

        return node_pair

    def _create_cell_graph(self) -> Graph:
        """
        Creates the graph of the cell with all the nodes and edges.
        Does not assign the operations on the edges or nodes.

        Returns:
            Graph representing the Cell.
        """

        cell = Graph()
        cell.name = "cell"

        cell.add_node(1)  # Input node

        for i in range(2, 12, 2):
            cell.add_node(i, comb_op=truncate_add) # Node for summation
            cell.add_node(i+1, subgraph=self._create_node_pair_graph(i+1)) # Node with node_pair as subgraph

        cell.add_node(12, comb_op=channel_concat)
        cell.add_node(13) # Output node
        cell.set_scope("cell", recursively=False)

        # Add edges
        cell.add_edges_densly()

        # Remove unnecessary edges
        cell.remove_edge(1, 12)

        edges = list(cell.edges())
        for u, v in edges:
            # We want to retain the edge from input to output, always
            if u == 1 and v == 13:
                continue
            # Remove edges from summation nodes (even nodes) to nodes other than its immediate neighbour
            # Remove edges to nodes with node_pair subgraph (odd nodes) which are not from their summation nodes
            elif (u%2 == 0 and v != u+1) or (v%2 == 1 and v != u+1):
                cell.remove_edge(u, v)

        return cell


    def _create_cell(self, cell: Graph, in_channels: int, out_channels: Union[int, None]=None,) -> Graph:
        """
        Creates a single cell used in the NASBench101 search space.

        Args:
            in_channels     : Number of input channels to the cell
            out_channels    : Number of output channels of the cell

        Returns:
            Graph representation of NASBench101 Cell.
        """
        if out_channels == None:
            out_channels = 2*in_channels

        dense_graph_matrix = np.triu(np.ones((7, 7)), 1) # Matrix representing densely connected cell space
        node_channels = compute_vertex_channels(in_channels, out_channels, dense_graph_matrix)

        # Update the node-pair graph inside each node so that their edges have the operations
        # (conv3x3, conv1x1, maxpool3x3) with the correct number of channels
        cell.update_nodes(
            update_func=lambda node, in_edges, out_edges: _set_cell_node_pair_ops(node, node_channels),
            scope="cell"
        )

        # The edges of the cell have Zero or Identity as the operations
        cell.update_edges(
            update_func=lambda edge: _set_cell_edge_ops(edge, node_channels=node_channels),
            scope="cell",
            private_edge_data=True,
        )

        return cell

    def convert_to_cell(self, matrix, ops):

        if len(matrix) < 7:
            # the nasbench spec can have an adjacency matrix of n x n for n<7, 
            # but in the nasbench api, it is always 7x7 (possibly containing blank rows)
            # so this method will add a blank row/column

            new_matrix = np.zeros((7, 7), dtype='int8')
            new_ops = []
            n = matrix.shape[0]
            for i in range(7):
                for j in range(7):
                    if j < n - 1 and i < n:
                        new_matrix[i][j] = matrix[i][j]
                    elif j == n - 1 and i < n:
                        new_matrix[i][-1] = matrix[i][j]

            for i in range(7):
                if i < n - 1:
                    new_ops.append(ops[i])
                elif i < 6:
                    new_ops.append('conv3x3-bn-relu')
                else:
                    new_ops.append('output')
            return {
                'matrix': new_matrix,
                'ops': new_ops
            }

        else:
            return {
                'matrix': matrix,
                'ops': ops
            }

    def query(
        self,
        metric=None,
        dataset="cifar10",
        path=None,
        epoch=-1,
        full_lc=False,
        dataset_api=None,
    ):
        """
        Query results from nasbench 101
        """
        assert isinstance(metric, Metric)
        assert dataset in ["cifar10", None], "Unknown dataset: {}".format(dataset)
        if metric in [Metric.ALL, Metric.HP]:
            raise NotImplementedError()
        if dataset_api is None:
            raise NotImplementedError("Must pass in dataset_api to query nasbench101")
        assert epoch in [
            -1,
            4,
            12,
            36,
            108,
            None,
        ], "nasbench101 does not have full learning curve info"

        metric_to_nb101 = {
            Metric.TRAIN_ACCURACY: "train_accuracy",
            Metric.VAL_ACCURACY: "validation_accuracy",
            Metric.TEST_ACCURACY: "test_accuracy",
            Metric.TRAIN_TIME: "training_time",
            Metric.PARAMETERS: "trainable_parameters",
        }

        if self.get_spec() is None:
            raise NotImplementedError(
                "Cannot yet query directly from the naslib object"
            )
        api_spec = dataset_api["api"].ModelSpec(**self.spec)

        if not dataset_api["nb101_data"].is_valid(api_spec):
            return -1

        query_results = dataset_api["nb101_data"].query(api_spec)
        if full_lc:
            vals = [
                dataset_api["nb101_data"].query(api_spec, epochs=e)[
                    metric_to_nb101[metric]
                ]
                for e in [4, 12, 36, 108]
            ]
            # return a learning curve with unique values only at 4, 12, 36, 108
            nums = [4, 8, 20, 56]
            lc = [val for i, val in enumerate(vals) for _ in range(nums[i])]
            if epoch == -1:
                return lc
            else:
                return lc[:epoch]

        if metric == Metric.RAW:
            return query_results
        elif metric == Metric.TRAIN_TIME:
            return query_results[metric_to_nb101[metric]]
        else:
            return query_results[metric_to_nb101[metric]] * 100

    def get_spec(self):
        if self.spec is None:
            self.spec = convert_naslib_to_spec(self)
        return self.spec

    def get_hash(self):
        return convert_spec_to_tuple(self.get_spec())

    def set_spec(self, spec, dataset_api=None):

        if isinstance(spec, str):
            """
            TODO: I couldn't find a better solution here.
            We need the arch iterator to return strings because the matrix/ops
            representation is too large for 400k elements. But having the `spec' be 
            strings would require passing in dataset_api for all of this search 
            space's methods. So the solution is to optionally pass in the dataset 
            api in set_spec and check whether `spec' is a string or a dict.
            """
            fix, comp = dataset_api["nb101_data"].get_metrics_from_hash(spec)
            spec = self.convert_to_cell(fix['module_adjacency'], fix['module_operations'])
            self.set_spec(spec)        
        self.spec = spec

    def get_arch_iterator(self, dataset_api=None):        
        return dataset_api["nb101_data"].hash_iterator()

    def sample_random_architecture(self, dataset_api):
        """
        This will sample a random architecture and update the edges in the
        naslib object accordingly.
        From the NASBench repository:
        one-hot adjacency matrix
        draw [0,1] for each slot in the adjacency matrix
        """
        while True:
            matrix = np.random.choice([0, 1], size=(NUM_VERTICES, NUM_VERTICES))
            matrix = np.triu(matrix, 1)
            ops = np.random.choice(OPS, size=NUM_VERTICES).tolist()
            ops[0] = INPUT
            ops[-1] = OUTPUT
            spec = dataset_api["api"].ModelSpec(matrix=matrix, ops=ops)
            if dataset_api["nb101_data"].is_valid(spec):
                break

        self.set_spec({"matrix": matrix, "ops": ops})

    def mutate(self, parent, dataset_api, edits=1):
        """
        This will mutate the parent architecture spec.
        Code inspird by https://github.com/google-research/nasbench
        """
        parent_spec = parent.get_spec()
        spec = copy.deepcopy(parent_spec)
        matrix, ops = spec['matrix'], spec['ops']
        for _ in range(edits):
            while True:
                new_matrix = copy.deepcopy(matrix)
                new_ops = copy.deepcopy(ops)
                for src in range(0, NUM_VERTICES - 1):
                    for dst in range(src+1, NUM_VERTICES):
                        if np.random.random() < 1 / NUM_VERTICES:
                            new_matrix[src][dst] = 1 - new_matrix[src][dst]
                for ind in range(1, NUM_VERTICES - 1):
                    if np.random.random() < 1 / len(OPS):
                        available = [op for op in OPS if op != new_ops[ind]]
                        new_ops[ind] = np.random.choice(available)
                new_spec = dataset_api['api'].ModelSpec(new_matrix, new_ops)
                if dataset_api['nb101_data'].is_valid(new_spec):
                    break
        
        self.set_spec({'matrix':new_matrix, 'ops':new_ops})

    def get_nbhd(self, dataset_api=None):
        # return all neighbors of the architecture
        spec = self.get_spec()
        matrix, ops = spec["matrix"], spec["ops"]
        nbhd = []

        def add_to_nbhd(new_matrix, new_ops, nbhd):
            new_spec = {"matrix": new_matrix, "ops": new_ops}
            model_spec = dataset_api["api"].ModelSpec(new_matrix, new_ops)
            if dataset_api["nb101_data"].is_valid(model_spec):
                nbr = NasBench101SearchSpace()
                nbr.set_spec(new_spec)
                nbr_model = torch.nn.Module()
                nbr_model.arch = nbr
                nbhd.append(nbr_model)
            return nbhd

        # add op neighbors
        for vertex in range(1, OP_SPOTS + 1):
            if is_valid_vertex(matrix, vertex):
                available = [op for op in OPS if op != ops[vertex]]
                for op in available:
                    new_matrix = copy.deepcopy(matrix)
                    new_ops = copy.deepcopy(ops)
                    new_ops[vertex] = op
                    nbhd = add_to_nbhd(new_matrix, new_ops, nbhd)

        # add edge neighbors
        for src in range(0, NUM_VERTICES - 1):
            for dst in range(src + 1, NUM_VERTICES):
                new_matrix = copy.deepcopy(matrix)
                new_ops = copy.deepcopy(ops)
                new_matrix[src][dst] = 1 - new_matrix[src][dst]
                new_spec = {"matrix": new_matrix, "ops": new_ops}

                if matrix[src][dst] and is_valid_edge(matrix, (src, dst)):
                    nbhd = add_to_nbhd(new_matrix, new_ops, nbhd)

                if not matrix[src][dst] and is_valid_edge(new_matrix, (src, dst)):
                    nbhd = add_to_nbhd(new_matrix, new_ops, nbhd)

        random.shuffle(nbhd)
        return nbhd

    def get_type(self):
        return "nasbench101"


def _set_node_pair_edge_op(edge, C):
    edge.data.set(
        "op",
        [
            ops.ConvBnReLU(C, C, kernel_size=3),
            ops.ConvBnReLU(C, C, kernel_size=1),
            ops.MaxPool(C, kernel_size=3, stride=1, use_bn=False),
        ]
    )


def _set_cell_node_pair_ops(node, node_channels):
    node_idx, node_data = node
    # Get the node-pair graph saved in this node, and updates its (only) edge
    # with the operations (conv3x3, conv1x1, maxpool3x3)

    if "subgraph" in node_data:
        channels = node_channels[(node_idx - 1)//2]
        node_pair = node_data["subgraph"]
        node_pair.update_edges(
            update_func=lambda edge: _set_node_pair_edge_op(edge, C=channels),
            scope="node_pair",
            private_edge_data=True
        )


def _set_cell_edge_ops(edge, node_channels: List[int]):
    if edge.head == 1:      # if this is an edge from the input node, then we have to apply a 1x1 projection
        C_in = node_channels[0]              # Number of input channels to cell
        C_out = node_channels[edge.tail//2]  # Number of channels expected by each node in the cell

        edge.data.set(
            "op",
            [
                ops.InputProjection(C_in, C_out, ops.Identity()),
                ops.InputProjection(C_in, C_out, ops.Zero(stride=1)),
            ],
        )
    elif edge.head%2==1:
        edge.data.set(
            "op",
            [
                ops.Identity(),
                ops.Zero(stride=1),
            ],
        )
    else:
        # Edges from even nodes are edges from summation nodes
        # They should always be Identity. No optimization of alpha required.
        edge.data.finalize()


def get_utilized(matrix):
    # return the sets of utilized edges and nodes
    # first, compute all paths
    n = np.shape(matrix)[0]
    sub_paths = []
    for j in range(0, n):
        sub_paths.append([[(0, j)]]) if matrix[0][j] else sub_paths.append([])

    # create paths sequentially
    for i in range(1, n - 1):
        for j in range(1, n):
            if matrix[i][j]:
                for sub_path in sub_paths[i]:
                    sub_paths[j].append([*sub_path, (i, j)])
    paths = sub_paths[-1]

    utilized_edges = []
    for path in paths:
        for edge in path:
            if edge not in utilized_edges:
                utilized_edges.append(edge)

    utilized_nodes = []
    for i in range(NUM_VERTICES):
        for edge in utilized_edges:
            if i in edge and i not in utilized_nodes:
                utilized_nodes.append(i)

    return utilized_edges, utilized_nodes


def num_edges_and_vertices(matrix):
    # return the true number of edges and vertices
    edges, nodes = self.get_utilized(matrix)
    return len(edges), len(nodes)


def is_valid_vertex(matrix, vertex):
    edges, nodes = get_utilized(matrix)
    return vertex in nodes


def is_valid_edge(matrix, edge):
    edges, nodes = get_utilized(matrix)
    return edge in edges

def compute_vertex_channels(input_channels, output_channels, matrix):
  """
  Taken from https://github.com/google-research/nasbench

  Computes the number of channels at every vertex.

  Given the input channels and output channels, this calculates the number of
  channels at each interior vertex. Interior vertices have the same number of
  channels as the max of the channels of the vertices it feeds into. The output
  channels are divided amongst the vertices that are directly connected to it.
  When the division is not even, some vertices may receive an extra channel to
  compensate.

  Args:
    input_channels: input channel count.
    output_channels: output channel count.
    matrix: adjacency matrix for the module (pruned by model_spec).

  Returns:
    list of channel counts, in order of the vertices.
  """
  num_vertices = np.shape(matrix)[0]

  vertex_channels = [0] * num_vertices
  vertex_channels[0] = input_channels
  vertex_channels[num_vertices - 1] = output_channels

  if num_vertices == 2:
    # Edge case where module only has input and output vertices
    return vertex_channels

  # Compute the in-degree ignoring input, axis 0 is the src vertex and axis 1 is
  # the dst vertex. Summing over 0 gives the in-degree count of each vertex.
  in_degree = np.sum(matrix[1:], axis=0, dtype=int)
  interior_channels = output_channels // in_degree[num_vertices - 1]
  correction = output_channels % in_degree[num_vertices - 1]  # Remainder to add

  # Set channels of vertices that flow directly to output
  for v in range(1, num_vertices - 1):
    if matrix[v, num_vertices - 1]:
      vertex_channels[v] = interior_channels
      if correction:
        vertex_channels[v] += 1
        correction -= 1

  # Set channels for all other vertices to the max of the out edges, going
  # backwards. (num_vertices - 2) index skipped because it only connects to
  # output.
  for v in range(num_vertices - 3, 0, -1):
    if not matrix[v, num_vertices - 1]:
      for dst in range(v + 1, num_vertices - 1):
        if matrix[v, dst]:
          vertex_channels[v] = max(vertex_channels[v], vertex_channels[dst])
    assert vertex_channels[v] > 0

  # Sanity check, verify that channels never increase and final channels add up.
  final_fan_in = 0
  for v in range(1, num_vertices - 1):
    if matrix[v, num_vertices - 1]:
      final_fan_in += vertex_channels[v]
    for dst in range(v + 1, num_vertices - 1):
      if matrix[v, dst]:
        assert vertex_channels[v] >= vertex_channels[dst]
  assert final_fan_in == output_channels or num_vertices == 2
  # num_vertices == 2 means only input/output nodes, so 0 fan-in

  return vertex_channels

def channel_concat(tensors):
    return torch.cat(tensors, dim=1)

def truncate_add(tensors):
    min_channels = min([t.shape[1] for t in tensors])
    return sum([t[:,:min_channels,:,:] for t in tensors])