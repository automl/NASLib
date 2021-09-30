import numpy as np

NASBENCH_MR_EMBEDDING_STRUCTURE = [4,
                                   8,
                                   10, 12, 16,
                                   18, 20, 22, 26, 30,
                                   32, 34, 36, 38, 42, 46, 50,
                                   52, 54, 56, 58, 60, 64, 68, 72,
                                   76]

ZERO = "Zero"
IDENTITY = "Identity"
OP_NAMES = [ZERO, IDENTITY]

def convert_naslib_to_op_indices(naslib_object):
    multi_resolution_cells = naslib_object._get_child_graphs()
    ops = []
    # edge_list = get_edge_list(multi_resolution_cells)
    edge_list = []
    for i, j in edge_list:
        ops.append(multi_resolution_cells.edges[i, j]["op"].get_op_name)
    return [OP_NAMES.index(name) for name in ops]

def convert_params_to_op_indices(embedding):
    str_of_embedding = ""
    for param in embedding:
        if param <= 4:
            encoding = 2
        else:
            encoding = 4
            param = param / 8
        str_of_embedding += bin(int(param)-1)[2:].zfill(encoding)
    return [int(param) for param in str_of_embedding]


def decode_param(embedding_param):
    param = 0
    for bin_code in embedding_param:
        param = param * 2 + bin_code
    if len(embedding_param) > 2:
        channel_encoding = 8
    else:
        channel_encoding = 1
    return (param+1) * channel_encoding


def convert_op_idices_to_params(op_indicies):
    embedding = []
    encoded_embedding = np.split(op_indicies, NASBENCH_MR_EMBEDDING_STRUCTURE)
    for embedding_param in encoded_embedding:
        embedding.append(decode_param(embedding_param))
    return embedding
