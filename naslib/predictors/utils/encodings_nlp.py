import numpy as np
import logging

"""
These are the encoding methods for nas-bench-nlp.
The plan is to unify encodings across all search spaces.
Note: this has not been thoroughly tested yet.
"""

logger = logging.getLogger(__name__)


def encode_adj(compact, num=25):
    # add all the edges, but change the final index to 'num'
    last_idx = len(compact[1]) - 1
    assert last_idx <= num
    def extend(idx):
        if idx == last_idx:
            return num
        return idx 
    
    adj_matrix = np.zeros((num+1, num+1))
    for edge in compact[0]:
        adj_matrix[extend(edge[0]), extend(edge[1])] = 1
    flattened = [int(i) for i in adj_matrix.flatten()]
    
    # add one-hot ops and hidden states    
    ops = [*compact[1][:-1], *[0]*(num - last_idx), compact[1][-1]]
    ops_onehot = []
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


def encode_nlp(arch, encoding_type='path'):
    
    compact = arch.get_compact()

    if encoding_type == 'adjacency_one_hot':
        return encode_adj(compact=compact)
    
    elif encoding_type == 'compact':
        return compact

    else:
        print('{} is not yet implemented as an encoding type \
         for darts'.format(encoding_type))
        raise NotImplementedError()