
"""
There are three representations
'naslib': the NASBench201SearchSpace object
'op_indices': A list of six ints, which is the simplest representation
'arch_str': The string representation used in the original nasbench201 paper

This file currently has the following conversions:
naslib -> op_indices
op_indices -> naslib
naslib -> arch_str

Note: we could add more conversions, but this is all we need for now
"""

OP_NAMES = ['Identity', 'Zero', 'ReLUConvBN3x3', 'ReLUConvBN1x1', 'AvgPool1x1']
EDGE_LIST = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))


def convert_naslib_to_op_indices(naslib_object):

    cell = naslib_object._get_child_graphs(single_instances=True)[0]
    ops = []
    for i, j in EDGE_LIST:
        ops.append(cell.edges[i, j]['op'].get_op_name)

    return [OP_NAMES.index(name) for name in ops]


def convert_op_indices_to_naslib(op_indices, naslib_object):
    """
    Converts op indices to a naslib object
    input: op_indices (list of six ints)
    naslib_object is an empty NasBench201SearchSpace() object.
    Do not call this method with a naslib object that has already been 
    discretized (i.e., all edges have a single op).

    output: none, but the naslib object now has all edges set
    as in genotype.
    
    warning: this method will modify the edges in naslib_object.
    """
    
    # create a dictionary of edges to ops
    edge_op_dict = {}
    for i, index in enumerate(op_indices):
        edge_op_dict[EDGE_LIST[i]] = OP_NAMES[index]
    
    def add_op_index(edge):
        # function that adds the op index from the dictionary to each edge
        if (edge.head, edge.tail) in edge_op_dict:
            for i, op in enumerate(edge.data.op):
                if op.get_op_name == edge_op_dict[(edge.head, edge.tail)]:
                    index = i
                    break
            edge.data.set('op_index', index, shared=True)

    def update_ops(edge):
        # function that replaces the primitive ops at the edges with the one in op_index
        if isinstance(edge.data.op, list):
            primitives = edge.data.op
        else:
            primitives = edge.data.primitives

        edge.data.set('op', primitives[edge.data.op_index])
        edge.data.set('primitives', primitives)     # store for later use

    naslib_object.update_edges(
        add_op_index,
        scope=naslib_object.OPTIMIZER_SCOPE,
        private_edge_data=False
    )
    
    naslib_object.update_edges(
        update_ops, 
        scope=naslib_object.OPTIMIZER_SCOPE,
        private_edge_data=True
    )

def convert_naslib_to_str(naslib_object):
    """
    Converts naslib object to string representation.
    """
    
    ops_to_nb201 = {
        'AvgPool1x1': 'avg_pool_3x3',
        'ReLUConvBN1x1': 'nor_conv_1x1',
        'ReLUConvBN3x3': 'nor_conv_3x3',
        'Identity': 'skip_connect',
        'Zero': 'none',
    }

    cell = naslib_object.edges[2, 3].op
    edge_op_dict = {
        (i, j): ops_to_nb201[cell.edges[i, j]['op'].get_op_name] for i, j in cell.edges
    }
    op_edge_list = [
        '{}~{}'.format(edge_op_dict[(i, j)], i-1) for i, j in sorted(edge_op_dict, key=lambda x: x[1])
    ]

    return '|{}|+|{}|{}|+|{}|{}|{}|'.format(*op_edge_list)