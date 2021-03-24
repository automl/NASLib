import numpy as np
"""
'naslib': the NASBench101SearchSpace object
'spec': adjacency matrix + op list
"""

def convert_naslib_to_spec(naslib_object):
    
    matrix = np.triu(np.ones((7,7)), 1)

    ops_to_nb101 = {
            'MaxPool1x1': 'maxpool3x3',
            'ReLUConvBN1x1': 'conv1x1-bn-relu',
            'ReLUConvBN3x3': 'conv3x3-bn-relu',
        }

    ops_to_nb101_edges = {
        'Identity': 1,
        'Zero': 0,
    }

    num_vertices = 7
    ops = ['input'] * num_vertices
    ops[-1] = 'output'

    cell = naslib_object.edges[2, 3].op
    print('cell', cell)
    
    for i in range(1, 6):
        ops[i] = ops_to_nb101[cell.nodes[i+1]['subgraph'].edges[1, 2]['op'].get_op_name]
    
    for i, j in cell.edges:
        matrix[i-1][j-1] = ops_to_nb101_edges[cell.edges[i, j]['op'].get_op_name]
        
    return [matrix, ops]


def convert_spec_to_naslib(spec, naslib_object):
    # TODO: write this method similar to how it was written for nasbench201 and darts
    raise NotImplementedError('Cannot yet convert a spec to naslib object')

def convert_spec_to_tuple(spec):
    # convert the spec to a hashable type
    op_dict = ['input', 'output', 'maxpool3x3', 'conv1x1-bn-relu', 'conv3x3-bn-relu']
    
    matrix = spec['matrix'].flatten()
    ops = [op_dict.index(s) for s in spec['ops']]
    tup = tuple([*matrix, *ops])
    return tup