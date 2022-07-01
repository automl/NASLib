from naslib.search_spaces import (
    DartsSearchSpace,
    SimpleCellSearchSpace,
    NasBench201SearchSpace,
    NasBench101SearchSpace,
    HierarchicalSearchSpace,
)
import subprocess
import numpy as np


def count_parameters_in_MB(model):
    return np.sum(
        np.prod(v.size()) for name, v in model.named_parameters()
        if "auxiliary" not in name) / 1e6


from naslib.search_spaces.darts.conversions import convert_naslib_to_genotype
ss = NasBench101SearchSpace()
# print child graph embedding within a macrograph
#print(ss._get_child_graphs())
# Sample a fixed set of operations
#print(ss.parameters())
ss.sample_random_architecture()
ss.parse()
print(count_parameters_in_MB(ss))
ss = NasBench201SearchSpace()
#print( get_gpu_memory_map())
ss.sample_random_architecture()
ss.parse()
print(count_parameters_in_MB(ss))
# Convert to pytorch model
ss.parse()
#print(ss.modules_str())
