import numpy as np
import logging

"""
These are the encoding methods for nas-bench-asr.
The plan is to unify encodings across all search spaces.
Note: this has not been thoroughly tested yet.
"""

logger = logging.getLogger(__name__)


def encode_compact(compact):
    from naslib.search_spaces.nasbenchasr.conversions import flatten
    return flatten(compact)

def encode_compact_mr(op_indices):
    encoding = []
    chunk = []
    for idx, digit in enumerate(op_indices):
        if idx % 3 == 0 or idx == len(op_indices) - 1:
            if len(chunk) > 0:
                value = 0
                for j, num in enumerate(chunk):
                    value += num * (2**j)
                if value > 8:
                    print("Bug is here")
                encoding.append(value)
                chunk = []
        chunk.append(digit)
    return encoding


def encode_seminas_nasbenchmr(compact):
    dic = {
        "num_vertices": 27,
        "adjacency": np.identity(27, dtype=np.float32),
        "operations": encode_compact_mr(compact),
        "mask": np.array([i < 27 for i in range(27)], dtype=np.float32),
        "val_acc": 0.0,
    }
    return dic


def encode_mr(arch, encoding_type='path', max_nodes=3, accs=None):

    op_indices = arch.get_op_indices()

    if encoding_type == 'seminas':
        return encode_seminas_nasbenchmr(op_indices)

    elif encoding_type == 'compact':
        return op_indices

    else:
        print('{} is not yet implemented as an encoding type \
         for asr'.format(encoding_type))
        raise NotImplementedError()
