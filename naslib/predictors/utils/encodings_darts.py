import numpy as np
import logging

"""
These are the encoding methods for DARTS.
The plan is to unify encodings across all search spaces.
"""

logger = logging.getLogger(__name__)

OPS = ['max_pool_3x3',
       'avg_pool_3x3',
       'skip_connect',
       'sep_conv_3x3',
       'sep_conv_5x5',
       'dil_conv_3x3',
       'dil_conv_5x5'
       ]
NUM_VERTICES = 4
INPUT_1 = 'c_k-2'
INPUT_2 = 'c_k-1'
OUTPUT = 'c_k'


def get_paths(arch):
    """ return all paths from input to output """

    path_builder = [[[], [], [], []], [[], [], [], []]]
    paths = [[], []]

    for i, cell in enumerate(arch):
        for j in range(len(OPS)):
            if cell[j][0] == 0:
                path = [INPUT_1, OPS[cell[j][1]]]
                path_builder[i][j//2].append(path)
                paths[i].append(path)
            elif cell[j][0] == 1:
                path = [INPUT_2, OPS[cell[j][1]]]
                path_builder[i][j//2].append(path)
                paths[i].append(path)
            else:
                for path in path_builder[i][cell[j][0] - 2]:
                    path = [*path, OPS[cell[j][1]]]
                    path_builder[i][j//2].append(path)
                    paths[i].append(path)

    return paths

def get_path_indices(arch, long_paths=True):
    """
    compute the index of each path
    There are 4 * (8^0 + ... + 8^4) paths total
    If long_paths = False, we give a single boolean to all paths of
    size 4, so there are only 4 * (1 + 8^0 + ... + 8^3) paths
    """
    paths = get_paths(arch)
    normal_paths, reduce_paths = paths
    num_ops = len(OPS)
    """
    Compute the max number of paths per input per cell.
    Since there are two cells and two inputs per cell, 
    total paths = 4 * max_paths
    """

    max_paths = sum([num_ops ** i for i in range(NUM_VERTICES + 1)])    
    path_indices = []

    # set the base index based on the cell and the input
    for i, paths in enumerate((normal_paths, reduce_paths)):
        for path in paths:
            index = i * 2 * max_paths
            if path[0] == INPUT_2:
                index += max_paths

            # recursively compute the index of the path
            for j in range(NUM_VERTICES + 1):
                if j == len(path) - 1:
                    path_indices.append(index)
                    break
                elif j == (NUM_VERTICES - 1) and not long_paths:
                    path_indices.append(2 * (i + 1) * max_paths - 1)
                    break
                else:
                    index += num_ops ** j * (OPS.index(path[j + 1]) + 1)

    return tuple(path_indices)

def encode_paths(arch, cutoff=None):
    # output one-hot encoding of paths
    path_indices = get_path_indices(arch)
    num_ops = len(OPS)

    max_paths = sum([num_ops ** i for i in range(NUM_VERTICES + 1)])    

    path_encoding = np.zeros(4 * max_paths)
    for index in path_indices:
        path_encoding[index] = 1
    if cutoff:
        path_encoding = path_encoding[:cutoff]
    return path_encoding


def encode_adj(arch):
    matrices = []
    ops = []
    true_num_vertices = NUM_VERTICES + 3
    for cell in arch:
        matrix = np.zeros((true_num_vertices, true_num_vertices))
        op_list = []
        for i, edge in enumerate(cell):
            dest = i//2 + 2
            matrix[edge[0]][dest] = 1
            op_list.append(edge[1])
        for i in range(2, 6):
            matrix[i][-1] = 1
        matrices.append(matrix)
        ops.append(op_list)

    encoding = [*matrices[0].flatten(), *ops[0], *matrices[1].flatten(), *ops[1]]
    return np.array(encoding)


def encode_darts(arch, encoding_type='path'):
    
    compact = arch.get_compact()

    if encoding_type == 'path':
        return encode_paths(arch=compact)
    
    elif encoding_type == 'adjacency_one_hot':
        return encode_adj(arch=compact)

    else:
        print('{} is not yet implemented as an encoding type \
         for darts'.format(encoding_type))
        raise NotImplementedError()