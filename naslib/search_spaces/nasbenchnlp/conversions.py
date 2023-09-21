import sys
import copy

"""
'recipe': the original encoding from NAS-Bench-NLP
'compact': a compact representation used to store architectures (variable length)
'adjacency': one-hot adjacency + categorical ops + hidden states. 
             Currently this is used by the surrogate model (fixed length)
"""


def convert_recipe_to_compact(recipe):
    nodes = ['x', 'h_prev_0', 'h_prev_1', 'h_prev_2']
    op_dict = ['in', 'activation_sigm', 'activation_tanh', 'activation_leaky_relu', \
               'elementwise_sum', 'elementwise_prod', 'linear', 'blend']
    ops = [0, 0, 0, 0]
    edges = []
    hiddens = []

    for node in recipe.keys():
        if node not in nodes:
            nodes.append(node)
        else:
            print('node repeated?')
            sys.exit()
        op_idx = op_dict.index(recipe[node]['op'])
        ops.append(op_idx)
        idx = nodes.index(node)

        if 'h_new' in node:
            hiddens.append(idx)

        for parent in recipe[node]['input']:
            if parent not in nodes:
                # reorder and call again
                new_recipe = copy.deepcopy(recipe)
                temp = new_recipe.pop(node)
                new_recipe[node] = temp
                return convert_recipe_to_compact(new_recipe)
            else:
                # add edge
                parent_idx = nodes.index(parent)
                edges.append((parent_idx, idx))

    return tuple(edges), tuple(ops), tuple(hiddens)


def convert_compact_to_recipe(compact):
    nodes = ['x', 'h_prev_0', 'h_prev_1', 'h_prev_2']
    op_dict = ['in', 'activation_sigm', 'activation_tanh', 'activation_leaky_relu',
               'elementwise_sum', 'elementwise_prod', 'linear', 'blend']

    edges, ops, hiddens = compact
    max_node_idx = max([max(edge) for edge in edges])

    # create the set of node names
    reg_node_idx = 0
    hidden_node_idx = 0
    for i in range(len(nodes), max_node_idx + 1):
        if i not in hiddens:
            nodes.append('node_{}'.format(reg_node_idx))
            reg_node_idx += 1
        else:
            nodes.append('h_new_{}'.format(hidden_node_idx))
            hidden_node_idx += 1

    recipe = {}
    for i in range(4, len(nodes)):
        node_dict = {}
        node_dict['op'] = op_dict[ops[i]]
        inputs = []
        for edge in edges:
            if edge[1] == i:
                inputs.append(nodes[edge[0]])
        node_dict['input'] = inputs
        recipe[nodes[i]] = node_dict

    return recipe


def make_compact_mutable(compact):
    # convert tuple to list so that it is mutable
    edge_list = []
    for edge in compact[0]:
        edge_list.append(list(edge))
    return [edge_list, list(compact[1]), list(compact[2])]
