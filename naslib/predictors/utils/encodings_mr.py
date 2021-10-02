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


def encode_mr(arch, encoding_type='path', max_nodes=3, accs=None):

    op_indices = arch.get_op_indices()

    if encoding_type == 'seminas':
        raise NotImplementedError("NAO/SemiNAS encoding not yet done for nas-bench-mr")

    elif encoding_type == 'compact':
        return op_indices

    else:
        print('{} is not yet implemented as an encoding type \
         for asr'.format(encoding_type))
        raise NotImplementedError()
