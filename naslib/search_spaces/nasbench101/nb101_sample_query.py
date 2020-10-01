from nasbench import api
import copy
import numpy as np
import matplotlib.pyplot as plt
import random
import os
# from naslib.utils.utils import get_project_root

# note: nb101 needs TF1.x for running as it uses multiple deprecated functions in
# API call.

get_project_root = '/home/mehtay/research/NASLib/naslib'
# Useful constants
INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NUM_VERTICES = 7
MAX_EDGES = 9
EDGE_SPOTS = NUM_VERTICES * (NUM_VERTICES - 1) / 2   # Upper triangular matrix
OP_SPOTS = NUM_VERTICES - 2   # Input/output vertices are fixed
ALLOWED_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
ALLOWED_EDGES = [0, 1]   # Binary adjacency matrix


nb101_datadir = os.path.join(get_project_root, 'data', 'nasbench_only108.tfrecord')

nasbench = api.NASBench(nb101_datadir)

matrix = np.triu(np.ones((7,7)), 1)

cell = api.ModelSpec(
  # matrix=[[0, 1, 1, 1, 0, 1, 0],    # input layer
  #         [0, 0, 0, 0, 0, 0, 1],    # 1x1 conv
  #         [0, 0, 0, 0, 0, 0, 1],    # 3x3 conv
  #         [0, 0, 0, 0, 1, 0, 0],    # 5x5 conv (replaced by two 3x3's)
  #         [0, 0, 0, 0, 0, 0, 1],    # 5x5 conv (replaced by two 3x3's)
  #         [0, 0, 0, 0, 0, 0, 1],    # 3x3 max-pool
  #         [0, 0, 0, 0, 0, 0, 0]],   # output layer
  # Operations at the vertices of the module, matches order of matrix.
  matrix = matrix,
  ops=[INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT])

# Querying multiple times may yield different results. Each cell is evaluated 3
# times at each epoch budget and querying will sample one randomly.
data = nasbench.query(cell)
for k, v in data.items():
  print('%s: %s' % (k, str(v)))