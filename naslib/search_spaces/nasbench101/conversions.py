import numpy as np

from archai.algos.nasbench101 import model_builder
from naslib.search_spaces.nasbench101.primitives import ModelWrapper

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NUM_VERTICES = 7

all_ops = ["input", "output", "maxpool3x3", "conv1x1-bn-relu", "conv3x3-bn-relu"]

def convert_spec_to_model(spec):
    model = model_builder.build(spec['matrix'], spec['ops'])
    model_wrapper = ModelWrapper(model)
    return model_wrapper

def convert_spec_to_tuple(spec):
    matrix = spec["matrix"].flatten()
    ops = [all_ops.index(s) for s in spec["ops"]]
    tup = tuple([*matrix, *ops])

    return tup

def convert_tuple_to_spec(tup):

    matrix_vals = tup[:-NUM_VERTICES]
    matrix = np.array(matrix_vals).reshape(NUM_VERTICES, NUM_VERTICES)
    ops = [all_ops[t] for t in tup[-NUM_VERTICES:]]

    return {"matrix": matrix, "ops": ops}
