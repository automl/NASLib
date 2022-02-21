import numpy as np
import logging

from naslib.search_spaces.nasbenchasr.conversions import flatten

"""
These are the encoding methods for nas-bench-asr.
The plan is to unify encodings across all search spaces.
Note: this has not been thoroughly tested yet.
"""

logger = logging.getLogger(__name__)


one_hot_ops = [
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
]


def encode_compact(compact):
    return flatten(compact)


def encode_adjacency_one_hot(compact):
    one_hot = []
    for e in flatten(compact):
        one_hot = [*one_hot, *one_hot_ops[e]]
    return one_hot


def encode_seminas_nasbenchasr(compact):
    # note: the adjacency matrix is fixed for ASR, 
    # so the identity matrix can be passed in
    dic = {
        "num_vertices": 9,
        "adjacency": np.identity(9, dtype=np.float32),
        "operations": flatten(compact),
        "mask": np.array([i < 9 for i in range(9)], dtype=np.float32),
        "val_acc": 0.0,
    }
    return dic


def encode_asr(arch, encoding_type='adjacency_one_hot', max_nodes=3, accs=None):

    compact = arch.get_compact()

    if encoding_type == 'adjacency_one_hot':
        return encode_adjacency_one_hot(compact)

    elif encoding_type == 'compact':
        return encode_compact(compact)

    elif encoding_type == 'seminas':
        return encode_seminas_nasbenchasr(compact)

    else:
        print('{} is not yet implemented as an encoding type \
         for asr'.format(encoding_type))
        raise NotImplementedError()
