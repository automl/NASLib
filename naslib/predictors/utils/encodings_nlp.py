import numpy as np
import logging

"""
These are the encoding methods for nas-bench-nlp.
The plan is to unify encodings across all search spaces.
Note: this has not been thoroughly tested yet.
"""

logger = logging.getLogger(__name__)


def get_adj_matrix(compact, num=25):
    # this method returns the flattened adjacency matrix only
    # 'num' is the maximum number of nodes in the search space
    last_idx = len(compact[1]) - 1
    assert last_idx <= num
    def extend(idx):
        if idx == last_idx:
            return num
        return idx 
    
    adj_matrix = np.zeros((num+1, num+1))
    for edge in compact[0]:
        adj_matrix[extend(edge[0]), extend(edge[1])] = 1
    
    return adj_matrix

def get_categorical_ops(compact, num=25):
    """
    This returns the set of ops, extended to account for the
    max number of nodes in the search space, so that it's the
    same size for all ops.
    """
    last_idx = len(compact[1]) - 1
    assert last_idx <= num
    return [*compact[1][:-1], *[0]*(num - last_idx), compact[1][-1]]

def encode_adj(compact, num=25):
    """
    this method returns the adjacency one hot encoding,
    which is a flattened adjacency matrix + one hot op encoding
    + flag for is_hidden_state on each node.
    'num' is the maximum number of nodes in the search space.
    """
    adj_matrix = get_adj_matrix(compact, num=num)
    flattened = [int(i) for i in adj_matrix.flatten()]

    # add one-hot ops and hidden states
    ops = get_categorical_ops(compact, num=num)
    ops_onehot = []
    last_idx = len(compact[1]) - 1
    for i, op in enumerate(ops):
        onehot = [1 if op == i else 0 for i in range(8)]
        ops_onehot.extend(onehot)
        if i in compact[2]:
            ops_onehot.append(1)
        elif i == num and last_idx in compact[2]:
            ops_onehot.append(1)
        else:
            ops_onehot.append(0)

    return np.array([*flattened, *ops_onehot])

def encode_seminas(compact, num=25):
    """
    note: this is temporary. This will be removed during the code cleanup
    note: there's no way to add the hidden node flag    
    """
    matrix = get_adj_matrix(compact, num=num)
    ops = get_categorical_ops(compact, num=num)
    # offset ops list by one
    ops = [op+1 for op in ops]

    dic = {
        'num_vertices': num,
        'adjacency': matrix,
        'operations': ops,
        'mask': np.array([i < num for i in range(num)], dtype=np.float32),
        'val_acc': 0.0
    }
    return dic

def encode_gcn(compact, num=25):
    '''
    note: this is temporary. This will be removed during the code cleanup
    '''
    matrix = get_adj_matrix(compact, num=num)
    matrix = np.array(matrix, dtype=np.float32)
    ops = get_categorical_ops(compact, num=num)
    op_map = [i for i in range(8)]
    ops_onehot = np.array([[i == op_map.index(op) for i in range(len(op_map))] for op in ops], dtype=np.float32)

    dic = {
        'num_vertices': num,
        'adjacency': matrix,
        'operations': ops_onehot,
        'mask': np.array([i < num for i in range(num)], dtype=np.float32),
        'val_acc': 0.0
    }
    return dic

def encode_nlp(arch, encoding_type='path', num=25):
    # 'num' is the maximum number of nodes in the search space

    compact = arch.get_compact()

    if encoding_type == 'adjacency_one_hot':
        return encode_adj(compact=compact, num=num)

    elif encoding_type == 'seminas':
        return encode_seminas(compact=compact, num=num)

    elif encoding_type == 'gcn':
        return encode_gcn(compact=compact, num=num)

    elif encoding_type == 'compact':
        return compact

    else:
        print('{} is not yet implemented as an encoding type \
         for nlp'.format(encoding_type))
        raise NotImplementedError()