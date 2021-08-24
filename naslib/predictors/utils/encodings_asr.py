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


def encode_asr(arch, encoding_type='path', max_nodes=3, accs=None):

    compact = arch.get_compact()

    if encoding_type == 'adjacency_one_hot':
        return encode_adj(compact=compact, max_nodes=max_nodes, one_hot=True)

    elif encoding_type == 'compact':
        return encode_compact(compact)

    else:
        print('{} is not yet implemented as an encoding type \
         for asr'.format(encoding_type))
        raise NotImplementedError()
