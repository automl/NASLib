import logging

from naslib.utils.encodings import EncodingType
from naslib.search_spaces.nasbench101.encodings import encode_101_spec
from naslib.search_spaces.nasbench201.encodings import encode_adjacency_one_hot_op_indices
from naslib.search_spaces.nasbench301.encodings import encode_darts_compact
from naslib.search_spaces.transbench101.encodings import encode_adjacency_one_hot_transbench_micro_op_indices, \
    encode_adjacency_one_hot_transbench_macro_op_indices
from naslib.search_spaces.nasbench101.conversions import convert_tuple_to_spec

"""
Currently we need search space specific methods.
The plan is to unify encodings across all search spaces.
nasbench201 and darts are implemented so far.
TODO: clean up this file.
"""

logger = logging.getLogger(__name__)


def encode_spec(spec, encoding_type=EncodingType.ADJACENCY_ONE_HOT, ss_type=None):
    if ss_type == 'nasbench101':
        if isinstance(spec, tuple):
            spec = convert_tuple_to_spec(spec)
        return encode_101_spec(spec, encoding_type=encoding_type)
    elif ss_type == 'nasbench201' and encoding_type == EncodingType.ADJACENCY_ONE_HOT:
        return encode_adjacency_one_hot_op_indices(spec)
    elif ss_type == 'nasbench301':
        return encode_darts_compact(spec, encoding_type=encoding_type)
    elif ss_type == 'transbench101_micro' and encoding_type == EncodingType.ADJACENCY_ONE_HOT:
        return encode_adjacency_one_hot_transbench_micro_op_indices(spec)
    elif ss_type == 'transbench101_macro' and encoding_type == EncodingType.ADJACENCY_ONE_HOT:
        return encode_adjacency_one_hot_transbench_macro_op_indices(spec)
    else:
        raise NotImplementedError(f'No implementation found for encoding search space {ss_type} with {encoding_type}')
