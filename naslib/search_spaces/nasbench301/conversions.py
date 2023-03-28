from collections import namedtuple

Genotype = namedtuple("Genotype", "normal normal_concat reduce reduce_concat")

"""
NASLib uses four representations of darts architectures:
'naslib': the NasBench301SearchSpace object
'genotype': representation used in the original DARTS paper
'compact': representation used in [Li and Talwalkar] and BANANAS (smallest size)
'config': representation used in nasbench301, based on ConfigSpace

This file contains all 12 types of conversions from one represenation to another.
"""


def get_cell_of_type(naslib_object, cell_type):
    for node in naslib_object.nodes:
        if (
                "subgraph" in naslib_object.nodes[node]
                and naslib_object.nodes[node]["subgraph"].name == cell_type
        ):
            return naslib_object.nodes[node]["subgraph"]

    raise Exception(f"Cell of type {cell_type} not found in the graph")


def convert_naslib_to_genotype(naslib_object):
    """convert the naslib representation to Genotype"""
    ops_to_genotype = {
        "Identity": "skip_connect",
        "FactorizedReduce": "skip_connect",
        "SepConv3x3": "sep_conv_3x3",
        "DilConv3x3": "dil_conv_3x3",
        "SepConv5x5": "sep_conv_5x5",
        "DilConv5x5": "dil_conv_5x5",
        "AvgPool": "avg_pool_3x3",
        "MaxPool": "max_pool_3x3",
        "Zero": "zero",
    }
    cells = [
        get_cell_of_type(naslib_object, "normal_cell"),
        get_cell_of_type(naslib_object, "reduction_cell"),
    ]
    converted_cells = []
    for cell in cells:
        edge_op_dict = {
            (i, j): ops_to_genotype[cell.edges[i, j]["op"].get_op_name]
            for i, j in cell.edges
        }
        op_edge_list = [
            (edge_op_dict[(i, j)], i - 1)
            for i, j in sorted(edge_op_dict, key=lambda x: x[1])
            if j < 7
        ]
        converted_cells.append(op_edge_list)

    return Genotype(
        normal=converted_cells[0],
        normal_concat=[2, 3, 4, 5],
        reduce=converted_cells[1],
        reduce_concat=[2, 3, 4, 5],
    )


def convert_genotype_to_naslib(genotype, naslib_object):
    """
    Converts the genotype representation to a naslib object
    input: genotype is the genotype representation
    naslib_object is an empty NasBench301SearchSpace() object.
    Do not call this method with a naslib object that has already been
    discretized (i.e., all but 2 incoming edges for each node are pruned).

    output: none, but the naslib object now has all edges set
    as in genotype.

    warning: this method will delete and modify the edges in naslib_object.
    """
    genotype_to_ops = {
        "skip_connect": ("Identity", "FactorizedReduce"),
        "sep_conv_3x3": "SepConv3x3",
        "dil_conv_3x3": "DilConv3x3",
        "sep_conv_5x5": "SepConv5x5",
        "dil_conv_5x5": "DilConv5x5",
        "avg_pool_3x3": "AvgPool",
        "max_pool_3x3": "MaxPool",
        # "zero": ("Zero"),
    }
    cell_names = ["normal_cell", "reduction_cell"]

    # create a dictionary of edges to ops in the genotype
    edge_op_dict = {"normal_cell": {}, "reduction_cell": {}}
    for c, cell_type in enumerate(["normal", "reduce"]):
        cell = eval("genotype." + cell_type)
        tail = 2
        for i, edge in enumerate(cell):
            if i % 2 == 0:
                tail += 1
            head = edge[1] + 1
            edge_op_dict[cell_names[c]][(head, tail)] = genotype_to_ops[edge[0]]

    def add_genotype_op_index(edge):
        # function that adds the op index from genotype to each edge, and deletes the rest
        if (edge.head, edge.tail) in edge_op_dict[edge.data.cell_name]:
            for i, op in enumerate(edge.data.op):
                if (
                        op.get_op_name
                        in edge_op_dict[edge.data.cell_name][(edge.head, edge.tail)]
                ):
                    index = i
                    break
            edge.data.set("op_index", index, shared=True)
        else:
            edge.data.delete()

    def update_ops(edge):
        # function that replaces the primitive ops at the edges with the ones from genotype
        if isinstance(edge.data.op, list):
            primitives = edge.data.op
        else:
            primitives = edge.data.primitives

        edge.data.set("op", primitives[edge.data.op_index])
        edge.data.set("primitives", primitives)  # store for later use

    naslib_object.update_edges(
        add_genotype_op_index,
        scope=naslib_object.OPTIMIZER_SCOPE,
        private_edge_data=False,
    )

    naslib_object.update_edges(
        update_ops, scope=naslib_object.OPTIMIZER_SCOPE, private_edge_data=True
    )


def convert_genotype_to_config(genotype):
    """Converts a DARTS genotype to a configspace instance dictionary"""
    base_string = "NetworkSelectorDatasetInfo:darts:"
    config = {}

    for cell_type in ["normal", "reduce"]:
        cell = eval("genotype." + cell_type)

        start = 0
        n = 2
        for node_idx in range(4):
            end = start + n
            ops = cell[2 * node_idx: 2 * node_idx + 2]

            # get edge idx
            edges = {
                base_string + "edge_" + cell_type + "_" + str(start + i): op
                for op, i in ops
            }
            config.update(edges)

            if node_idx != 0:
                # get node idx
                input_nodes = sorted(list(map(lambda x: x[1], ops)))
                input_nodes_idx = "_".join([str(i) for i in input_nodes])
                config.update(
                    {
                        base_string
                        + "inputs_node_"
                        + cell_type
                        + "_"
                        + str(node_idx + 2): input_nodes_idx
                    }
                )

            start = end
            n += 1
    return config


def convert_config_to_genotype(config):
    """Converts a configspace instance dictionary to a DARTS genotype"""
    base_string = "NetworkSelectorDatasetInfo:darts:"
    genotype = []
    for i, cell_type in enumerate(["normal", "reduce"]):
        genotype.append([])

        start = 0
        n = 2
        for node_idx in range(4):
            end = start + n
            # print(start, end)
            for j in range(start, end):
                key = "NetworkSelectorDatasetInfo:darts:edge_{}_{}".format(cell_type, j)
                if key in config:
                    genotype[i].append((config[key], j - start))

            if len(genotype[i]) != 2 * (node_idx + 1):
                print("this is not a valid darts arch")
                return config

            start = end
            n += 1

    return Genotype(
        normal=genotype[0],
        normal_concat=[2, 3, 4, 5],
        reduce=genotype[1],
        reduce_concat=[2, 3, 4, 5],
    )


def convert_genotype_to_compact(genotype):
    """Converts Genotype to the compact representation"""
    OPS = [
        "max_pool_3x3",
        "avg_pool_3x3",
        "skip_connect",
        "sep_conv_3x3",
        "sep_conv_5x5",
        "dil_conv_3x3",
        "dil_conv_5x5",
    ]
    compact = []

    for i, cell_type in enumerate(["normal", "reduce"]):
        cell = eval("genotype." + cell_type)
        compact.append([])

        for j in range(8):
            compact[i].append((cell[j][1], OPS.index(cell[j][0])))

    compact_tuple = (tuple(compact[0]), tuple(compact[1]))
    return compact_tuple


def convert_compact_to_genotype(compact):
    """Converts the compact representation to a Genotype"""
    OPS = [
        "max_pool_3x3",
        "avg_pool_3x3",
        "skip_connect",
        "sep_conv_3x3",
        "sep_conv_5x5",
        "dil_conv_3x3",
        "dil_conv_5x5",
    ]
    genotype = []

    for i in range(2):
        cell = compact[i]
        genotype.append([])

        for j in range(8):
            genotype[i].append((OPS[cell[j][1]], cell[j][0]))

    return Genotype(
        normal=genotype[0],
        normal_concat=[2, 3, 4, 5],
        reduce=genotype[1],
        reduce_concat=[2, 3, 4, 5],
    )


def make_compact_mutable(compact):
    # convert tuple to list so that it is mutable
    arch_list = []
    for cell in compact:
        arch_list.append([])
        for pair in cell:
            arch_list[-1].append([])
            for num in pair:
                arch_list[-1][-1].append(num)
    return arch_list


def make_compact_immutable(compact):
    # convert list to tuple so that it is hashable
    arch_list = []
    for cell in compact:
        arch_list.append([])
        for pair in cell:
            arch_list[-1].append(tuple(pair))
        arch_list[-1] = tuple(arch_list[-1])
    return tuple(arch_list)


def convert_naslib_to_config(naslib_object):
    genotype = convert_naslib_to_genotype(naslib_object)
    return convert_genotype_to_config(genotype)


def convert_config_to_naslib(config, naslib_object):
    genotype = convert_config_to_genotype(config)
    return convert_genotype_to_naslib(genotype, naslib_object)


def convert_naslib_to_compact(naslib_object):
    genotype = convert_naslib_to_genotype(naslib_object)
    return convert_genotype_to_compact(genotype)


def convert_compact_to_naslib(compact, naslib_object):
    genotype = convert_compact_to_genotype(compact)
    return convert_genotype_to_naslib(genotype, naslib_object)


def convert_config_to_compact(config):
    genotype = convert_config_to_genotype(config)
    return convert_genotype_to_compact(genotype)


def convert_compact_to_config(compact):
    genotype = convert_compact_to_genotype(compact)
    return convert_genotype_to_config(genotype)
