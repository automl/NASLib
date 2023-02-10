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

OPS = ["avg_pool_3x3", "nor_conv_1x1", "nor_conv_3x3", "none", "skip_connect"]
NUM_OPS = len(OPS)


def get_paths(arch):
    """
    return all paths from input to output
    """
    ops = arch.get_op_indices()
    path_blueprints = [[3], [0, 4], [1, 5], [0, 2, 5]]
    paths = []
    for blueprint in path_blueprints:
        paths.append([ops[node] for node in blueprint])
    return paths


def get_path_indices(arch, num_ops=5):
    """
    compute the index of each path
    """
    paths = get_paths(arch)
    path_indices = []

    for i, path in enumerate(paths):
        if i == 0:
            index = 0
        elif i in [1, 2]:
            index = num_ops
        else:
            index = num_ops + num_ops ** 2
        for j, op in enumerate(path):
            index += op * num_ops ** j
        path_indices.append(index)

    return tuple(path_indices)


def encode_adjacency_one_hot(arch):
    encoding = arch.get_op_indices()
    one_hot = []
    for e in encoding:
        one_hot = [*one_hot, *one_hot_nasbench201[e]]
    return one_hot


def encode_paths(arch, num_ops=5, longest_path_length=3):
    """output one-hot encoding of paths"""
    num_paths = sum([num_ops ** i for i in range(1, longest_path_length + 1)])
    path_indices = get_path_indices(arch, num_ops=num_ops)
    encoding = np.zeros(num_paths)
    for index in path_indices:
        encoding[index] = 1
    return encoding


def encode_gcn_nasbench201(arch):
    """
    Input:
    a list of categorical ops starting from 0
    """
    ops = arch.get_op_indices()
    # offset ops list by one, add input and output to ops list
    ops = [op + 1 for op in ops]
    ops = [0, *ops, 6]
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


def add_global_node(mx, ifAdj):
    """add a global node to operation or adjacency matrixs, fill diagonal for adj and transpose adjs"""
    if ifAdj:
        mx = np.column_stack((mx, np.ones(mx.shape[0], dtype=np.float32)))
        mx = np.row_stack((mx, np.zeros(mx.shape[1], dtype=np.float32)))
        np.fill_diagonal(mx, 1)
        mx = mx.T
    else:
        mx = np.column_stack((mx, np.zeros(mx.shape[0], dtype=np.float32)))
        mx = np.row_stack((mx, np.zeros(mx.shape[1], dtype=np.float32)))
        mx[mx.shape[0] - 1][mx.shape[1] - 1] = 1
    return mx


def encode_bonas_nasbench201(arch):
    """
    Input:
    a list of categorical ops starting from 0
    """
    ops = arch.get_op_indices()
    # offset ops list by one, add input and output to ops list
    ops = [op + 1 for op in ops]
    ops = [0, *ops, 6]
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

    matrix = add_global_node(matrix, True)
    ops_onehot = add_global_node(ops_onehot, False)

    matrix = np.array(matrix, dtype=np.float32)
    ops_onehot = np.array(ops_onehot, dtype=np.float32)

    dic = {"adjacency": matrix, "operations": ops_onehot, "val_acc": 0.0}
    return dic


def encode_seminas_nasbench201(arch):
    """
    Input:
    a list of categorical ops starting from 0
    """
    ops = arch.get_op_indices()
    # offset ops list by one, add input and output to ops list
    ops = [op + 1 for op in ops]
    ops = [0, *ops, 6]
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


def encode_201(arch, encoding_type=EncodingType.ADJACENCY_ONE_HOT):
    if encoding_type == EncodingType.ADJACENCY_ONE_HOT:
        return encode_adjacency_one_hot(arch)

    elif encoding_type == EncodingType.PATH:
        return encode_paths(arch)

    elif encoding_type == EncodingType.GCN:
        return encode_gcn_nasbench201(arch)

    elif encoding_type == EncodingType.BONAS:
        return encode_bonas_nasbench201(arch)

    elif encoding_type == EncodingType.SEMINAS:
        return encode_seminas_nasbench201(arch)

    else:
        logger.info(f"{encoding_type} is not yet supported as an architecture encoding for nb201")
        raise NotImplementedError()


def encode_adjacency_one_hot_op_indices(op_indices):
    one_hot = []
    for e in op_indices:
        one_hot = [*one_hot, *one_hot_nasbench201[e]]
    return one_hot


def encode_spec(spec, encoding_type=EncodingType.ADJACENCY_ONE_HOT):
    if encoding_type == EncodingType.ADJACENCY_ONE_HOT:
        return encode_adjacency_one_hot_op_indices(spec)
    else:
        raise NotImplementedError(f'No implementation found for encoding search space nb201 with {encoding_type}')
