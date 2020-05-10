import torch

from naslib.search_spaces.nasbench_201.primitives import NAS_BENCH_201


def discretize_architectural_weights(arch_weights):
    arch_weight_idx_to_parent = {0: 0,
                                 1: 0,
                                 2: 1,
                                 3: 0,
                                 4: 1,
                                 5: 2}
    arch_strs = {
        'cell_normal_from_0_to_1': '',
        'cell_normal_from_0_to_2': '',
        'cell_normal_from_0_to_3': '',
        'cell_normal_from_1_to_2': '',
        'cell_normal_from_1_to_3': '',
        'cell_normal_from_2_to_3': '',
    }
    for arch_weight_idx, (edge_key, edge_weights) in enumerate(arch_weights.items()):
        edge_weights_norm = torch.softmax(edge_weights, dim=-1)
        selected_op_str = NAS_BENCH_201[edge_weights_norm.argmax()]
        arch_strs[edge_key] = '|{}~{}|'.format(selected_op_str, arch_weight_idx_to_parent[arch_weight_idx])

    arch_str = arch_strs['cell_normal_from_0_to_1'] + '+' \
               + arch_strs['cell_normal_from_0_to_2'] + arch_strs['cell_normal_from_1_to_2'] + '+' \
               + arch_strs['cell_normal_from_0_to_3'] + arch_strs['cell_normal_from_1_to_3'] + \
               arch_strs['cell_normal_from_2_to_3']
