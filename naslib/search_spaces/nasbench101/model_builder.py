from typing import List
import logging

import torch
from torch import nn

from .model import Network
from .model_spec import ModelSpec

from archai.common import ml_utils

EXAMPLE_VERTEX_OPS = ['input', 'conv1x1-bn-relu', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3', 'output']

EXAMPLE_DESC_MATRIX = [[0, 1, 1, 1, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0]]

def build(desc_matrix:List[List[int]], vertex_ops:List[str], device=None,
          stem_out_channels=128, num_stacks=3, num_modules_per_stack=3, num_labels=10)->nn.Module:
    model_spec = ModelSpec(desc_matrix, vertex_ops)
    model = Network(model_spec, stem_out_channels, num_stacks, num_modules_per_stack, num_labels)
    logging.info(f'Model parameters: {ml_utils.param_size(model)}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
    model.to(device)
    return model