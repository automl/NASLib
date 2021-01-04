
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

    for i in range(1, 6):
        ops[i] = ops_to_nb101[cell.nodes[i+1]['subgraph'].edges[1, 2]['op'].get_op_name]
    
    for i, j in cell.edges:
        matrix[i-1][j-1] = ops_to_nb101_edges[cell.edges[i, j]['op'].get_op_name]
        
    return [matrix, ops]
