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

    print('matrix 1:\n {}\n matrix 2:\n {}\n'.format(matrices[0],matrices[1]))

    encoding = [*matrices[0].flatten(), *ops[0], *matrices[1].flatten(), *ops[1]]
    return np.array(encoding)

def encode_bonas(arch):
    matrices = []
    ops = []
    # true_num_vertices = NUM_VERTICES + 3
    for cell in arch:
        mat,op = transform_matrix(cell)
        matrices.append(mat)
        ops.append(op)

    mat_length = len(matrices[0][0])
    merged_length = len(matrices[0][0])*2
    matrix_merged = np.zeros((merged_length,merged_length))

    for col in range(mat_length):
        for row in range(col):
            matrix_merged[row,col] = matrices[0][row,col]
            matrix_merged[row+mat_length,col+mat_length] = matrices[1][row,col]

    matrix_final = add_global_node(matrix_merged,True)
    ops = np.concatenate((ops[0],ops[1]),axis=0)

    # print('architectures:')
    # print(arch)
    # print('matrix 1:\n {}\n matrix 2:\n {}\n'.format(matrices[0],matrices[1]))
    # print('matrix merged: \n {}\n'.format(matrix_merged))
    # print('matrix added: \n {}\n'.format(matrix_final))
    # print('ops:{}'.format(ops))
    ops_onehot = add_global_node(ops,False)
    #     matrix = np.zeros((true_num_vertices, true_num_vertices))
    #     op_list = []
    #     for i, edge in enumerate(cell):
    #         dest = i//2 + 2
    #         matrix[edge[0]][dest] = 1
    #         op_list.append(edge[1])
    #     for i in range(2, 6):
    #         matrix[i][-1] = 1
    #     matrices.append(matrix)
    #     ops.append(op_list)
    # # Merging two adj matrices, big adj mat should have dim 12 x 12
    # true_num_vertices_merged = NUM_VERTICES*4 + 6
    # offset = true_num_vertices_merged//2
    # print('architectures:')
    # print(arch)
    # print('matrix 1:\n {}\n matrix 2:\n {}\n'.format(matrices[0],matrices[1]))
    # for cell in arch:
    #     matrix_merged = np.zeros((true_num_vertices_merged, true_num_vertices_merged)) #np.eye(true_num_vertices)
        
    #     op_list = []
    #     for col in range(0,true_num_vertices):
    #         for row in range(0,col):
    #             #print('row:{}, col:{}\n'.format(row,col))
    #             matrix_merged[2*row,2*col]=matrices[0][row,col]
    #             matrix_merged[2*row+1,2*col]=matrices[0][row,col]
    #             matrix_merged[2*row+offset,col+offset]=matrices[1][row,col]
    #             matrix_merged[2*row+offset+1,col+offset]=matrices[1][row,col]

    # print('matrix merged: \n {}\n'.format(matrix_merged))
    # matrix_final = add_global_node(matrix_merged,True)
    # #print('matrix added: \n {}\n'.format(matrix_final))
    # print('ops:{}'.format(ops))
    # # list of real ops (input , OPS list, concat, output)
    # ops_onehot = np.zeros((true_num_vertices_merged+1,7+3))
    # for i in range(len(ops[0])):
    #     #offset 1, account for input op
    #     idx1 = ops[0][i]+1
    #     idx2 = ops[1][i]+1
    #     ops_onehot[i+2,idx1] = 1.
    #     ops_onehot[i+2+offset,idx2] = 1.

    # ops_onehot[0:2,0] = 1.
    # ops_onehot[offset-1,-2] = 1.
    # ops_onehot[offset:offset+2,0] = 1.
    # ops_onehot[true_num_vertices_merged-1,-2] = 1.
    # ops_onehot[-1,-1] = 1.
    #print('matrix 1:\n {}\n matrix 2:\n {}\n'.format(matrices[0],matrices[1]))
    print('ops one hot:\n{}'.format(ops_onehot))
    matrix_final = np.array(matrix_final,dtype=np.float32)
    ops_onehot = np.array(ops_onehot,dtype=np.float32)
    dic = {
        'adjacency': matrix_final,
        'operations': ops_onehot,
        'val_acc': 0.0
    }
    return dic

def add_global_node( mx, ifAdj):
    """add a global node to operation or adjacency matrixs, fill diagonal for adj and transpose adjs"""
    if (ifAdj):
        mx = np.column_stack((mx, np.ones(mx.shape[0], dtype=np.float32)))
        mx = np.row_stack((mx, np.zeros(mx.shape[1], dtype=np.float32)))
        np.fill_diagonal(mx, 1)
        mx = mx.T
    else:
        mx = np.column_stack((mx, np.zeros(mx.shape[0], dtype=np.float32)))
        mx = np.row_stack((mx, np.zeros(mx.shape[1], dtype=np.float32)))
        mx[mx.shape[0] - 1][mx.shape[1] - 1] = 1
    return mx

def transform_matrix(cell):
    normal = cell

    node_num = len(normal)+3

    adj = np.zeros((node_num, node_num))

    ops = np.zeros((node_num, len(OPS)+2))

    for i in range(len(normal)):
        connect, op = normal[i]
        if connect == 0 or connect==1:
            adj[connect][i+2] = 1
        else:
            adj[(connect-2)*2+2][i+2] = 1
            adj[(connect-2)*2+3][i+2] = 1
        ops[i+2][op] = 1
    adj[2:-1, -1] = 1
    ops[0:2, 0] = 1
    ops[-1][-1] = 1
    return adj, ops



def encode_darts(arch, encoding_type='path'):
    
    compact = arch.get_compact()

    if encoding_type == 'path':
        return encode_paths(arch=compact)
    
    elif encoding_type == 'adjacency_one_hot':
        return encode_adj(arch=compact)
    
    elif encoding_type == 'compact':
        return compact
    
    elif encoding_type == 'bonas':
        return encode_bonas(arch=compact)

    else:
        print('{} is not yet implemented as an encoding type \
         for darts'.format(encoding_type))
        raise NotImplementedError()