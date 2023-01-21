import logging

from enum import Enum
from naslib.search_spaces.nasbenchnlp.encodings import encode_nlp
from naslib.search_spaces.nasbench101.encodings import encode_101, encode_101_spec
from naslib.search_spaces.nasbench201.encodings import encode_adjacency_one_hot_op_indices, encode_201
from naslib.search_spaces.nasbench301.encodings import encode_darts, encode_darts_compact
from naslib.search_spaces.nasbenchasr.encodings import encode_asr
from naslib.search_spaces.transbench101.encodings import encode_tb101, \
    encode_adjacency_one_hot_transbench_micro_op_indices, encode_adjacency_one_hot_transbench_macro_op_indices
from naslib.search_spaces.nasbench101.conversions import convert_tuple_to_spec

"""
Currently we need search space specific methods.
The plan is to unify encodings across all search spaces.
nasbench201 and darts are implemented so far.
TODO: clean up this file.
"""

logger = logging.getLogger(__name__)


def encode(arch, encoding_type="adjacency_one_hot", ss_type=None):
    # this method calls either encode_201 or encode_darts based on the search space

    if ss_type == "nasbench101":
        return encode_101(arch, encoding_type=encoding_type)
    elif ss_type == "nasbench201":
        return encode_201(arch, encoding_type=encoding_type)
    elif ss_type == "nasbench301":
        return encode_darts(arch, encoding_type=encoding_type)
    elif ss_type == "nlp":
        return encode_nlp(arch,
                          encoding_type=encoding_type,
                          max_nodes=12,
                          accs=None)
    elif ss_type == 'transbench101_micro' or 'transbench101_macro':
        return encode_tb101(arch, encoding_type=encoding_type)
    elif ss_type == "asr":
        return encode_asr(arch, encoding_type=encoding_type)
    else:
        raise NotImplementedError(
            "{} is not yet supported for encodings".format(ss_type)
        )


def encode_spec(spec, encoding_type='adjacency_one_hot', ss_type=None):
    if ss_type == 'nasbench101':
        if isinstance(spec, tuple):
            spec = convert_tuple_to_spec(spec)
        return encode_101_spec(spec, encoding_type=encoding_type)
    elif ss_type == 'nasbench201' and encoding_type == 'adjacency_one_hot':
        return encode_adjacency_one_hot_op_indices(spec)
    elif ss_type == 'nasbench301':
        return encode_darts_compact(spec, encoding_type=encoding_type)
    elif ss_type == 'transbench101_micro' and encoding_type == 'adjacency_one_hot':
        return encode_adjacency_one_hot_transbench_micro_op_indices(spec)
    elif ss_type == 'transbench101_macro' and encoding_type == 'adjacency_one_hot':
        return encode_adjacency_one_hot_transbench_macro_op_indices(spec)
    else:
        raise NotImplementedError(f'No implementation found for encoding search space {ss_type} with {encoding_type}')
