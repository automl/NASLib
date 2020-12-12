from collections import namedtuple


Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


def convert_genotype_to_config(arch):
    """Converts a DARTS genotype to a configspace instance dictionary"""
    base_string = 'NetworkSelectorDatasetInfo:darts:'
    config = {}

    for cell_type in ['normal', 'reduce']:
        cell = eval('arch.' + cell_type)

        start = 0
        n = 2
        for node_idx in range(4):
            end = start + n
            ops = cell[2 * node_idx: 2 * node_idx + 2]

            # get edge idx
            edges = {base_string + 'edge_' + cell_type + '_' + str(start + i): op for
                         op, i in ops}
            config.update(edges)

            if node_idx != 0:
                # get node idx
                input_nodes = sorted(list(map(lambda x: x[1], ops)))
                input_nodes_idx = '_'.join([str(i) for i in input_nodes])
                config.update({base_string + 'inputs_node_' + cell_type + '_' + str(node_idx + 2):
                                   input_nodes_idx})

            start = end
            n += 1
    return config

def convert_config_to_genotype(config):
    """Converts a configspace instance dictionary to a DARTS genotype"""
    base_string = 'NetworkSelectorDatasetInfo:darts:'
    genotype = []
    for i, cell_type in enumerate(['normal', 'reduce']):
        genotype.append([])
        
        start = 0
        n = 2
        for node_idx in range(4):
            end = start + n
            #print(start, end)
            for j in range(start, end):
                key = 'NetworkSelectorDatasetInfo:darts:edge_{}_{}'.format(cell_type, j)
                if key in config:
                    genotype[i].append((config[key], j - start))
                    
            if len(genotype[i]) != 2 * (node_idx + 1):
                print('this is not a valid darts arch')
                return config
            
            start = end
            n += 1

    return Genotype(
        normal = genotype[0],
        normal_concat = [2, 3, 4, 5, 6],
        reduce = genotype[1],
        reduce_concat = [4, 5, 6]    
    )


def convert_genotype_to_compact(genotype):
    """ 
    Converts Genotype to the compact representation 
    used in [Li and Talwalkar 2018] and BANANAS
    """
    OPS = ['none',
           'max_pool_3x3',
           'avg_pool_3x3',
           'skip_connect',
           'sep_conv_3x3',
           'sep_conv_5x5',
           'dil_conv_3x3',
           'dil_conv_5x5'
           ]
    compact = []

    for i, cell_type in enumerate(['normal', 'reduce']):
        cell = eval('genotype.' + cell_type)
        compact.append([])
        
        for j in range(8):
            compact[i].append((cell[j][1], OPS.index(cell[j][0])))
        
    compact_tuple = (tuple(compact[0]), tuple(compact[1]))
    return compact_tuple
    
def convert_compact_to_genotype(compact):
    """ 
    Converts the compact representation used in 
    [Li and Talwalkar 2018] and BANANAS, to a Genotype
    """
    OPS = ['none',
           'max_pool_3x3',
           'avg_pool_3x3',
           'skip_connect',
           'sep_conv_3x3',
           'sep_conv_5x5',
           'dil_conv_3x3',
           'dil_conv_5x5'
           ]
    genotype = []

    for i in range(2):
        cell = compact[i]
        genotype.append([])
        
        for j in range(8):
            genotype[i].append((OPS[cell[j][1]], cell[j][0]))
        
    return Genotype(
        normal = genotype[0],
        normal_concat = [2, 3, 4, 5, 6],
        reduce = genotype[1],
        reduce_concat = [4, 5, 6]    
    )