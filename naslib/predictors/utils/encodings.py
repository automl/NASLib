import numpy as np
import logging

from naslib.predictors.utils.encodings_darts import encode_darts

"""
Currently we need search space specific methods.
The plan is to unify encodings across all search spaces.
nasbench201 and darts are implemented so far.
TODO: clean up this file.
"""

logger = logging.getLogger(__name__)


one_hot_nasbench201 = [[1,0,0,0,0],
                       [0,1,0,0,0],
                       [0,0,1,0,0],
                       [0,0,0,1,0],
                       [0,0,0,0,1]]

OPS = ['avg_pool_3x3', 'nor_conv_1x1', 'nor_conv_3x3', 'none', 'skip_connect']
NUM_OPS = len(OPS)


def encode_adjacency_categorical(arch):
    encoding = []
    cells = arch._get_child_graphs(single_instances=True)

    for cell in cells:
        edges = [(u, v) for u, v, data in sorted(cell.edges(data=True)) if not data.is_final()]
        for edge in edges:
            encoding.append(cell.edges[edge].op_index)
            
        return encoding

    
def encode_adjacency_one_hot(arch):
    
    encoding = encode_adjacency_categorical(arch)
    one_hot = []
    for e in encoding:
        one_hot = [*one_hot, *one_hot_nasbench201[e]]
    return one_hot
    

def get_op_indices(arch):

    cells = arch._get_child_graphs(single_instances=True)
    op_indices = []
    for cell in cells:
        edges = [(u, v) for u, v, data in sorted(cell.edges(data=True)) if not data.is_final()]
    for edge in edges:
        op_indices.append(cell.edges[edge].op_index)
        
    return op_indices


def get_paths(arch):
    """ 
    return all paths from input to output
    """
    path_blueprints = [[3], [0,4], [1,5], [0,2,5]]
    ops = get_op_indices(arch)
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


def encode_paths(arch, num_ops=5, longest_path_length=3):
    """ output one-hot encoding of paths """
    num_paths = sum([num_ops ** i for i in range(1, longest_path_length + 1)])
    path_indices = get_path_indices(arch, num_ops=num_ops)
    encoding = np.zeros(num_paths)
    for index in path_indices:
        encoding[index] = 1
    return encoding


def encode_gcn_nasbench201(arch):
    '''
    Input:
    a list of categorical ops starting from 0
    '''
    ops = encode_adjacency_categorical(arch)
    # offset ops list by one, add input and output to ops list
    ops = [op+1 for op in ops]
    ops = [0, *ops, 6]
    #print(ops)
    ops_onehot = np.array([[i == op for i in range(7)] for op in ops], dtype=np.float32)
    matrix = np.array(
            [[0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0]],dtype=np.float32)
    #matrix = np.transpose(matrix)
    dic = {
        'num_vertices': 8,
        'adjacency': matrix,
        'operations': ops_onehot,
        'mask': np.array([i < 8 for i in range(8)], dtype=np.float32),
        'val_acc': 0.0
    }

    return dic


def encode_bonas_nasbench201(arch):
    '''
    Input:
    a list of categorical ops starting from 0
    '''
    ops = encode_adjacency_categorical(arch)
    # offset ops list by one, add input and output to ops list
    ops = [op+1 for op in ops]
    ops = [0, *ops, 6]
    #print(ops)
    ops_onehot = np.array([[i == op for i in range(7)] for op in ops], dtype=np.float32)
    matrix = np.array(
            [[1, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 1]],dtype=np.float32)
    matrix = np.transpose(matrix)
    
    dic = {
        'adjacency': matrix,
        'operations': ops_onehot,
        'val_acc': 0.0
    }
    return dic

def encode_seminas_nasbench201(arch):
    '''
    Input:
    a list of categorical ops starting from 0
    '''
    ops = encode_adjacency_categorical(arch)
    # offset ops list by one, add input and output to ops list
    ops = [op+1 for op in ops]
    ops = [0, *ops, 6]
    #print(ops)
    #ops_onehot = np.array([[i == op for i in range(7)] for op in ops], dtype=np.float32)
    matrix = np.array(
            [[0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0]],dtype=np.float32)
    #matrix = np.transpose(matrix)
    dic = {
        'num_vertices': 8,
        'adjacency': matrix,
        'operations': ops,
        'mask': np.array([i < 8 for i in range(8)], dtype=np.float32),
        'val_acc': 0.0
    }

    return dic

def encode_201(arch, encoding_type='adjacency_one_hot'):
        
    if encoding_type == 'adjacency_one_hot':
        return encode_adjacency_one_hot(arch)
    
    elif encoding_type == 'path':
        return encode_paths(arch)

    elif encoding_type == 'gcn':
        return encode_gcn_nasbench201(arch)
    
    elif encoding_type == 'bonas':
        return encode_bonas_nasbench201(arch)

    elif encoding_type == 'seminas':
        return encode_seminas_nasbench201(arch)

    else:
        logger.info('{} is not yet supported as a predictor encoding'.format(encoding_type))
        raise NotImplementedError()

        
def encode(arch, encoding_type='adjacency_one_hot', ss_type='nasbench201'):
    # this method calls either encode_201 or encode_darts based on the search space

    if ss_type == 'nasbench201':
        return encode_201(arch, encoding_type=encoding_type)
    elif ss_type == 'darts':
        return encode_darts(arch, encoding_type=encoding_type)
    else:
        logger.info('{} is not yet supported for encodings'.format(ss_type))
        raise NotImplementedError()
