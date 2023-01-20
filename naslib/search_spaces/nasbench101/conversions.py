import numpy as np
import torch.nn as nn

from nasbench_pytorch.model import Network
from nasbench_pytorch.model import ModelSpec

from naslib.search_spaces.nasbench101.primitives import ModelWrapper

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

all_ops = [INPUT, OUTPUT, MAXPOOL3X3, CONV1X1, CONV3X3]


def get_children(model):
    children = list(model.children())
    all_children = []
    if children == []:
        return model
    else:
        for child in children:
            try:
                all_children.extend(get_children(child))
            except TypeError:
                all_children.append(get_children(child))

    return all_children


def convert_spec_to_model(spec):
    spec = ModelSpec(spec['matrix'], spec['ops'])
    model = Network(spec,
                    num_labels=10,
                    in_channels=3,
                    stem_out_channels=128,
                    num_stacks=3,
                    num_modules_per_stack=3)

    all_leaf_modules = get_children(model)
    inplace_relus = [module for module in all_leaf_modules if (isinstance(module, nn.ReLU) and module.inplace == True)]

    for relu in inplace_relus:
        relu.inplace = False

    model_wrapper = ModelWrapper(model)
    return model_wrapper


def convert_spec_to_tuple(spec):
    matrix = spec["matrix"].flatten()
    ops = [all_ops.index(s) for s in spec["ops"]]
    tup = tuple([*matrix, *ops])

    return tup


def convert_tuple_to_spec(tup):
    l = len(tup)
    # l = n*n + n
    n = int(-0.5 + np.sqrt(1 + 4 * l) / 2)
    matrix_vals = tup[:-n]
    matrix = np.array(matrix_vals).reshape(n, n)
    ops = [all_ops[t] for t in tup[-n:]]

    return {"matrix": matrix, "ops": ops}
