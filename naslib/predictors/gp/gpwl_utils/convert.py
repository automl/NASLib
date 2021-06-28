# Author: Xingchen Wan @ University of Oxford
# Ru, B., Wan, X., et al., 2021. "Interpretable Neural Architecture Search via Bayesian Optimisation using Weisfiler-Lehman Kernels". In ICLR 2021.

import numpy as np
import torch
import networkx as nx
import copy


def convert_n101_arch_to_graph(arch, prune_arch=True):
    from naslib.predictors.utils.encodings_nb101 import OPS_INCLUSIVE
    arch = arch.get_spec()
    matrix, ops = arch['matrix'], arch['ops']
    if prune_arch:
        matrix, ops = prune(matrix, ops)
    ops = [OPS_INCLUSIVE.index(op) for op in ops]
    g_nx = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
    for i, n in enumerate(ops):
        g_nx.nodes[i]['op_name'] = n
    return g_nx


def prune(original_matrix, ops):
    """Prune the extraneous parts of the graph.

    General procedure:
      1) Remove parts of graph not connected to input.
      2) Remove parts of graph not connected to output.
      3) Reorder the vertices so that they are consecutive after steps 1 and 2.

    These 3 steps can be combined by deleting the rows and columns of the
    vertices that are not reachable from both the input and output (in reverse).
    """
    num_vertices = np.shape(original_matrix)[0]
    new_matrix = copy.deepcopy(original_matrix)
    new_ops = copy.deepcopy(ops)
    # DFS forward from input
    visited_from_input = {0}
    frontier = [0]
    while frontier:
        top = frontier.pop()
        for v in range(top + 1, num_vertices):
            if original_matrix[top, v] and v not in visited_from_input:
                visited_from_input.add(v)
                frontier.append(v)

    # DFS backward from output
    visited_from_output = {num_vertices - 1}
    frontier = [num_vertices - 1]
    while frontier:
        top = frontier.pop()
        for v in range(0, top):
            if original_matrix[v, top] and v not in visited_from_output:
                visited_from_output.add(v)
                frontier.append(v)

    # Any vertex that isn't connected to both input and output is extraneous to
    # the computation graph.
    extraneous = set(range(num_vertices)).difference(
        visited_from_input.intersection(visited_from_output))

    # If the non-extraneous graph is less than 2 vertices, the input is not
    # connected to the output and the spec is invalid.
    if len(extraneous) > num_vertices - 2:
        new_matrix = None
        new_ops = None
        valid_spec = False
        return

    new_matrix = np.delete(new_matrix, list(extraneous), axis=0)
    new_matrix = np.delete(new_matrix, list(extraneous), axis=1)
    for index in sorted(extraneous, reverse=True):
        del new_ops[index]

    return new_matrix, new_ops


def convert_n201_arch_to_graph(arch_str):
    """Convert a nas-bench-201 string to a compatible networkx graph"""
    from naslib.search_spaces.nasbench201.conversions import convert_naslib_to_str
    OPS = ['input', 'avg_pool_3x3', 'nor_conv_1x1', 'nor_conv_3x3', 'none', 'skip_connect', 'output']

    # split the string into lists
    arch_str = convert_naslib_to_str(arch_str)
    arch_str_list = arch_str.split('|')
    ops = []
    for str_i in arch_str_list:
        if '~' in str_i:
            ops.append(str_i[:-2])

    G = nx.DiGraph()
    edge_list = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 6), (3, 6), (4, 7), (5, 7), (6, 7)]
    G.add_edges_from(edge_list)

    # assign node attributes and collate the information for nodes to be removed
    # (i.e. nodes with 'skip_connect' or 'none' label)
    node_labelling = ['input'] + ops + ['output']
    nodes_to_remove_list = []
    remove_nodes_list = []
    edges_to_add_list = []
    for i, n in enumerate(node_labelling):
        G.nodes[i]['op_name'] = n
        if n == 'none' or n == 'skip_connect':
            input_nodes = [edge[0] for edge in G.in_edges(i)]
            output_nodes = [edge[1] for edge in G.out_edges(i)]
            nodes_to_remove_info = {'id': i, 'input_nodes': input_nodes, 'output_nodes': output_nodes}
            nodes_to_remove_list.append(nodes_to_remove_info)
            remove_nodes_list.append(i)

            if n == 'skip_connect':
                for n_i in input_nodes:
                    edges_to_add = [(n_i, n_o) for n_o in output_nodes]
                    edges_to_add_list += edges_to_add

    # reconnect edges for removed nodes with 'skip_connect'
    G.add_edges_from(edges_to_add_list)

    # remove nodes with 'skip_connect' or 'none' label
    G.remove_nodes_from(remove_nodes_list)

    # after removal, some op nodes have no input nodes and some have no output nodes
    # --> remove these redundant nodes
    nodes_to_be_further_removed = []
    for n_id in G.nodes():
        in_edges = G.in_edges(n_id)
        out_edges = G.out_edges(n_id)
        if n_id != 0 and len(in_edges) == 0:
            nodes_to_be_further_removed.append(n_id)
        elif n_id != 7 and len(out_edges) == 0:
            nodes_to_be_further_removed.append(n_id)

    G.remove_nodes_from(nodes_to_be_further_removed)
    return G


def convert_darts_arch_to_graph(genotype, return_reduction=True, ):
    from naslib.search_spaces.darts.conversions import convert_naslib_to_genotype, Genotype

    genotype = convert_naslib_to_genotype(genotype)

    def _cell2graph(cell, concat):
        G = nx.DiGraph()
        n_nodes = (len(cell) // 2) * 3 + 3
        G.add_nodes_from(range(n_nodes), op_name=None)
        n_ops = len(cell) // 2
        G.nodes[0]['op_name'] = 'input1'
        G.nodes[1]['op_name'] = 'input2'
        G.nodes[n_nodes - 1]['op_name'] = 'output'
        for i in range(n_ops):
            G.nodes[i * 3 + 2]['op_name'] = cell[i * 2][0]
            G.nodes[i * 3 + 3]['op_name'] = cell[i * 2 + 1][0]
            G.nodes[i * 3 + 4]['op_name'] = 'add'
            G.add_edge(i * 3 + 2, i * 3 + 4)
            G.add_edge(i * 3 + 3, i * 3 + 4)

        for i in range(n_ops):
            # Add the connections to the input
            for offset in range(2):
                if cell[i * 2 + offset][1] == 0:
                    G.add_edge(0, i * 3 + 2 + offset)
                elif cell[i * 2 + offset][1] == 1:
                    G.add_edge(1, i * 3 + 2 + offset)
                else:
                    k = cell[i * 2 + offset][1] - 2
                    # Add a connection from the output of another block
                    G.add_edge(int(k) * 3 + 4, i * 3 + 2 + offset)
        # Add connections to the output
        for i in concat:
            if i <= 1:
                G.add_edge(i, n_nodes - 1)  # Directly from either input to the output
            else:
                op_number = i - 2
                G.add_edge(op_number * 3 + 4, n_nodes - 1)
        # If remove the skip link nodes, do another sweep of the graph
        for j in range(n_nodes):
            try:
                G.nodes[j]
            except KeyError:
                continue
            if G.nodes[j]['op_name'] == 'skip_connect':
                in_edges = list(G.in_edges(j))
                out_edge = list(G.out_edges(j))[0][1]  # There should be only one out edge really...
                for in_edge in in_edges:
                    G.add_edge(in_edge[0], out_edge)
                G.remove_node(j)
            elif G.nodes[j]['op_name'] == 'none':
                G.remove_node(j)
        for j in range(n_nodes):
            try:
                G.nodes[j]
            except KeyError:
                continue

            if G.nodes[j]['op_name'] not in ['input1', 'input2']:
                # excepting the input nodes, if the node has no incoming edge, remove it
                if len(list(G.in_edges(j))) == 0:
                    G.remove_node(j)
            elif G.nodes[j]['op_name'] != 'output':
                # excepting the output nodes, if the node has no outgoing edge, remove it
                if len(list(G.out_edges(j))) == 0:
                    G.remove_node(j)
            elif G.nodes[j]['op_name'] == 'add':  # If add has one incoming edge only, remove the node
                in_edges = list(G.in_edges(j))
                out_edges = list(G.out_edges(j))
                if len(in_edges) == 1 and len(out_edges) == 1:
                    G.add_edge(in_edges[0][0], out_edges[0][1])
                    G.remove_node(j)

        return G

    # todo: the naslib conversion gives normal_concat in [2,3,4,5,6]. Check whether that is alright?
    #   Here if I used [2,3,4,5,6] there will be errors in the graph conversion. So the code below is a temporary patch
    #   Xingchen Wan (Feb 2021)
    genotype = Genotype(
        normal=genotype.normal,
        normal_concat=[2, 3, 4, 5],
        reduce=genotype.reduce,
        reduce_concat=[2, 3, 4, 5]
    )

    G_normal = _cell2graph(genotype.normal, genotype.normal_concat)
    try:
        G_reduce = _cell2graph(genotype.reduce, genotype.reduce_concat)
    except AttributeError:
        G_reduce = None
    if return_reduction and G_reduce is not None:
        return G_normal, G_reduce
    else:
        return G_normal, None
