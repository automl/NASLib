import numpy as np

"""
Currently only implemented for NAS-Bench-201.
The plan is to make this work more broadly.
"""

one_hot_nasbench201 = [[1,0,0,0,0],
                       [0,1,0,0,0],
                       [0,0,1,0,0],
                       [0,0,0,1,0],
                       [0,0,0,0,1]]
OPS = ['avg_pool_3x3', 'nor_conv_1x1', 'nor_conv_3x3', 'none', 'skip_connect']
NUM_OPS = len(OPS)


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


def encode(arch, encoding_type='adjacency_one_hot'):
        
    encoding = []
    cells = arch._get_child_graphs(single_instances=True)

    for cell in cells:
        edges = [(u, v) for u, v, data in sorted(cell.edges(data=True)) if not data.is_final()]
        for edge in edges:
            encoding.append(cell.edges[edge].op_index)

    if encoding_type == 'adjacency_categorical':
        return encoding
    
    elif encoding_type == 'adjacency_one_hot':
        one_hot = []
        for e in encoding:
            one_hot = [*one_hot, *one_hot_nasbench201[e]]
        return one_hot
    elif encoding_type == 'path':
        return encode_paths(arch)
                
    else:
        logging.info('{} is not yet supported as a predictor encoding'.format(encoding_type))
        raise NotImplementedError()
