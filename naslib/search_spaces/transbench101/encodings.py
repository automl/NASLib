import logging

import numpy as np

from naslib.utils.encodings import EncodingType

logger = logging.getLogger(__name__)

one_hot_nasbench201 = [
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
]

one_hot_transnasbench201 = [[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]]

# not used at the moment
one_hot_transbench101 = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
]


def encode_adjacency_one_hot_tb101(arch):
    encoding = arch.get_op_indices()
    one_hot = []
    for e in encoding:
        one_hot = [*one_hot, *one_hot_nasbench201[e]]
    return one_hot


def encode_adjacency_one_hot_transbench_macro_op_indices(op_indices):
    one_hot = []
    one_hot_mapping = np.eye(5)

    if len(op_indices) < 6:
        op_indices = op_indices + tuple((0 for i in range(6 - len(op_indices))))

    for e in op_indices:
        one_hot = [*one_hot, *one_hot_mapping[e]]
    return one_hot


def encode_gcn_transbench101(arch):
    """
    Input:
    a list of categorical ops starting from 0
    """
    ops = arch.get_op_indices()
    # offset ops list by one, add input and output to ops list
    ops = [op + 1 for op in ops]
    ops = [0, *ops, 5]
    # print(ops)
    ops_onehot = np.array([[i == op for i in range(7)] for op in ops], dtype=np.float32)
    matrix = np.array(
        [
            [0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    # matrix = np.transpose(matrix)
    dic = {
        "num_vertices": 8,
        "adjacency": matrix,
        "operations": ops_onehot,
        "mask": np.array([i < 8 for i in range(8)], dtype=np.float32),
        "val_acc": 0.0,
    }

    return dic


def encode_seminas_transbench101(arch):
    """
    Input:
    a list of categorical ops starting from 0
    """
    ops = arch.get_op_indices()
    # offset ops list by one, add input and output to ops list
    ops = [op + 1 for op in ops]
    ops = [0, *ops, 5]
    matrix = np.array(
        [
            [0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    # matrix = np.transpose(matrix)
    dic = {
        "num_vertices": 8,
        "adjacency": matrix,
        "operations": ops,
        "mask": np.array([i < 8 for i in range(8)], dtype=np.float32),
        "val_acc": 0.0,
    }

    return dic


def encode_tb101(arch, encoding_type='adjacency_one_hot'):
    if encoding_type == EncodingType.ADJACENCY_ONE_HOT:
        return encode_adjacency_one_hot_tb101(arch)

    elif encoding_type == EncodingType.GCN:
        return encode_gcn_transbench101(arch)

    elif encoding_type == EncodingType.SEMINAS:
        return encode_seminas_transbench101(arch)

    else:
        logger.info(f"{encoding_type} is not yet supported as a predictor encoding for tnb101")
        raise NotImplementedError()


def encode_adjacency_one_hot_transbench_micro_op_indices(op_indices):
    one_hot = []
    for e in op_indices:
        one_hot = [*one_hot, *one_hot_transnasbench201[e]]
    return one_hot


def encode_adjacency_one_hot_transbench_micro(arch):
    encoding = arch.get_op_indices()
    return encode_adjacency_one_hot_transbench_micro_op_indices(encoding)


def encode_spec(spec, encoding_type=EncodingType.ADJACENCY_ONE_HOT, ss_type=None):
    if ss_type == 'transbench101_micro' and encoding_type == EncodingType.ADJACENCY_ONE_HOT:
        return encode_adjacency_one_hot_transbench_micro_op_indices(spec)
    elif ss_type == 'transbench101_macro' and encoding_type == EncodingType.ADJACENCY_ONE_HOT:
        return encode_adjacency_one_hot_transbench_macro_op_indices(spec)
    else:
        raise NotImplementedError(f'No implementation found for encoding search space {ss_type} with {encoding_type}')
