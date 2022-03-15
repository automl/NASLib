import sys
import copy
import numpy as np
import logging

"""
'recipe': the original encoding from NAS-Bench-NLP
'compact': a compact representation used to store architectures (variable length)
'adjacency': one-hot adjacency + categorical ops + hidden states. 
             Currently this is used by the surrogate model (fixed length)
"""
logger = logging.getLogger(__name__)


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

    return (tuple(edges), tuple(ops), tuple(hiddens))

def convert_compact_to_recipe(compact):
    nodes = ['x', 'h_prev_0', 'h_prev_1', 'h_prev_2']
    op_dict = ['in', 'activation_sigm', 'activation_tanh', 'activation_leaky_relu', \
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

def get_adj_matrix(compact, max_nodes):
    # this method returns the flattened adjacency matrix only
    last_idx = len(compact[1]) - 1
    assert last_idx <= max_nodes
    def extend(idx):
        if idx == last_idx:
            return max_nodes
        return idx 

    adj_matrix = np.zeros((max_nodes+1, max_nodes+1))
    for edge in compact[0]:
        adj_matrix[extend(edge[0]), extend(edge[1])] = 1

    return adj_matrix

def get_categorical_ops(compact, max_nodes):
    """
    This returns the set of ops, extended to account for the
    max number of nodes in the search space, so that it's the
    same size for all ops.
    """
    last_idx = len(compact[1]) - 1
    assert last_idx <= max_nodes
    return [*compact[1][:-1], *[0]*(max_nodes - last_idx), compact[1][-1]]

def get_categorical_hidden_states(compact, max_hidden_states=3):
    assert len(compact[2]) <= max_hidden_states
    return [*compact[2], *[0]*(max_hidden_states - len(compact[2]))]

def encode_adj(compact, max_nodes, one_hot=False, accs=None):
    """
    this method returns the adjacency one hot encoding,
    which is a flattened adjacency matrix + one hot op encoding
    + flag for is_hidden_state on each node.
    """
    adj_matrix = get_adj_matrix(compact, max_nodes=max_nodes)
    flattened = [int(i) for i in adj_matrix.flatten()]
    assert len(flattened) == (max_nodes + 1) ** 2

    # add ops and hidden states
    ops = get_categorical_ops(compact, max_nodes=max_nodes)
    assert len(ops) == max_nodes + 1
    hidden_states = get_categorical_hidden_states(compact)
    assert len(hidden_states) == 3
    if not one_hot:
        if accs is not None:
            return [*flattened, *ops, *hidden_states, *accs]
        return [*flattened, *ops, *hidden_states]

    ops_onehot = []
    last_idx = len(compact[1]) - 1
    for i, op in enumerate(ops):
        onehot = [1 if op == i else 0 for i in range(8)]
        ops_onehot.extend(onehot)
        if i in compact[2]:
            ops_onehot.append(1)
        elif i == max_nodes and last_idx in compact[2]:
            ops_onehot.append(1)
        else:
            ops_onehot.append(0)

    return [*flattened, *ops_onehot]

def encode_seminas(compact, max_nodes=25):
    """
    note: this is temporary. This will be removed during the code cleanup
    note: there's no way to add the hidden node flag    
    """
    matrix = get_adj_matrix(compact, max_nodes=max_nodes)
    ops = get_categorical_ops(compact, max_nodes=max_nodes)
    # offset ops list by one
    ops = [op+1 for op in ops]

    dic = {
        'num_vertices': max_nodes,
        'adjacency': matrix,
        'operations': ops,
        'mask': np.array([i < max_nodes for i in range(max_nodes)], dtype=np.float32),
        'val_acc': 0.0
    }
    return dic

def encode_gcn(compact, max_nodes=25):
    '''
    note: this is temporary. This will be removed during the code cleanup
    '''
    matrix = get_adj_matrix(compact, max_nodes=max_nodes)
    matrix = np.array(matrix, dtype=np.float32)
    ops = get_categorical_ops(compact, max_nodes=max_nodes)
    op_map = [i for i in range(8)]
    ops_onehot = np.array([[i == op_map.index(op) for i in range(len(op_map))] for op in ops], dtype=np.float32)

    dic = {
        'num_vertices': max_nodes,
        'adjacency': matrix,
        'operations': ops_onehot,
        'mask': np.array([i < max_nodes for i in range(max_nodes)], dtype=np.float32),
        'val_acc': 0.0
    }
    return dic

def encode_nlp(arch, encoding_type='path', max_nodes=25, accs=None):

    compact = arch.get_compact()

    if encoding_type == 'adjacency_one_hot':
        return encode_adj(compact=compact, max_nodes=max_nodes, one_hot=True)
    
    elif encoding_type == 'adjacency_mix':
        return encode_adj(compact=compact, max_nodes=max_nodes, one_hot=False, accs=accs)

    elif encoding_type == 'seminas':
        return encode_seminas(compact=compact, max_nodes=max_nodes)

    elif encoding_type == 'gcn':
        return encode_gcn(compact=compact, max_nodes=max_nodes)

    elif encoding_type == 'compact':
        return compact

    else:
        print('{} is not yet implemented as an encoding type \
         for nlp'.format(encoding_type))
        raise NotImplementedError()

