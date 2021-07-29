# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""Builds the Pytorch computational graph.
Tensors flowing into a single vertex are added together for all vertices
except the output, which is concatenated instead. Tensors flowing out of input
are always added.
If interior edge channels don't match, drop the extra channels (channels are
guaranteed non-decreasing). Tensors flowing out of the input as always
projected instead.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math

from .nasbench1_ops import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, spec, stem_out, num_stacks, num_mods, num_classes, bn=True):
        super(Network, self).__init__()

        self.spec = spec
        self.stem_out = stem_out
        self.num_stacks = num_stacks
        self.num_mods = num_mods
        self.num_classes = num_classes

        self.layers = nn.ModuleList([])

        in_channels = 3
        out_channels = stem_out

        # initial stem convolution
        stem_conv = ConvBnRelu(in_channels, out_channels, 3, 1, 1, bn=bn)
        self.layers.append(stem_conv)

        in_channels = out_channels
        for stack_num in range(num_stacks):
            if stack_num > 0:
                downsample = nn.MaxPool2d(kernel_size=2, stride=2)
                self.layers.append(downsample)

                out_channels *= 2

            for _ in range(num_mods):
                cell = Cell(spec, in_channels, out_channels, bn=bn)
                self.layers.append(cell)
                in_channels = out_channels

        self.classifier = nn.Linear(out_channels, num_classes)

        self._initialize_weights()

    def forward(self, x):
        for _, layer in enumerate(self.layers):
            x = layer(x)
        out = torch.mean(x, (2, 3))
        out = self.classifier(out)

        return out

    def get_prunable_copy(self, bn=False):

        model_new = Network(
            self.spec,
            self.stem_out,
            self.num_stacks,
            self.num_mods,
            self.num_classes,
            bn=bn,
        )

        # TODO this is quite brittle and doesn't work with nn.Sequential when bn is different
        # it is only required to maintain initialization -- maybe init after get_punable_copy?
        model_new.load_state_dict(self.state_dict(), strict=False)
        model_new.train()

        return model_new

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class Cell(nn.Module):
    """
    Builds the model using the adjacency matrix and op labels specified. Channels
    controls the module output channel count but the interior channels are
    determined via equally splitting the channel count whenever there is a
    concatenation of Tensors.
    """

    def __init__(self, spec, in_channels, out_channels, bn=True):
        super(Cell, self).__init__()

        self.spec = spec
        self.num_vertices = np.shape(self.spec.matrix)[0]

        # vertex_channels[i] = number of output channels of vertex i
        self.vertex_channels = ComputeVertexChannels(
            in_channels, out_channels, self.spec.matrix
        )
        # self.vertex_channels = [in_channels] + [out_channels] * (self.num_vertices - 1)

        # operation for each node
        self.vertex_op = nn.ModuleList([None])
        for t in range(1, self.num_vertices - 1):
            op = OP_MAP[spec.ops[t]](
                self.vertex_channels[t], self.vertex_channels[t], bn=bn
            )
            self.vertex_op.append(op)

        # operation for input on each vertex
        self.input_op = nn.ModuleList([None])
        for t in range(1, self.num_vertices):
            if self.spec.matrix[0, t]:
                self.input_op.append(
                    Projection(in_channels, self.vertex_channels[t], bn=bn)
                )
            else:
                self.input_op.append(None)

    def forward(self, x):
        tensors = [x]

        out_concat = []
        for t in range(1, self.num_vertices - 1):
            fan_in = [
                Truncate(tensors[src], self.vertex_channels[t])
                for src in range(1, t)
                if self.spec.matrix[src, t]
            ]

            if self.spec.matrix[0, t]:
                fan_in.append(self.input_op[t](x))

            # perform operation on node
            # vertex_input = torch.stack(fan_in, dim=0).sum(dim=0)
            vertex_input = sum(fan_in)
            # vertex_input = sum(fan_in) / len(fan_in)
            vertex_output = self.vertex_op[t](vertex_input)

            tensors.append(vertex_output)
            if self.spec.matrix[t, self.num_vertices - 1]:
                out_concat.append(tensors[t])

        if not out_concat:
            assert self.spec.matrix[0, self.num_vertices - 1]
            outputs = self.input_op[self.num_vertices - 1](tensors[0])
        else:
            if len(out_concat) == 1:
                outputs = out_concat[0]
            else:
                outputs = torch.cat(out_concat, 1)

            if self.spec.matrix[0, self.num_vertices - 1]:
                outputs += self.input_op[self.num_vertices - 1](tensors[0])

            # if self.spec.matrix[0, self.num_vertices-1]:
            #    out_concat.append(self.input_op[self.num_vertices-1](tensors[0]))
            # outputs = sum(out_concat) / len(out_concat)

        return outputs


def Projection(in_channels, out_channels, bn=True):
    """1x1 projection (as in ResNet) followed by batch normalization and ReLU."""
    return ConvBnRelu(in_channels, out_channels, 1, bn=bn)


def Truncate(inputs, channels):
    """Slice the inputs to channels if necessary."""
    input_channels = inputs.size()[1]
    if input_channels < channels:
        raise ValueError("input channel < output channels for truncate")
    elif input_channels == channels:
        return inputs  # No truncation necessary
    else:
        # Truncation should only be necessary when channel division leads to
        # vertices with +1 channels. The input vertex should always be projected to
        # the minimum channel count.
        assert input_channels - channels == 1
        return inputs[:, :channels, :, :]


def ComputeVertexChannels(in_channels, out_channels, matrix):
    """Computes the number of channels at every vertex.
    Given the input channels and output channels, this calculates the number of
    channels at each interior vertex. Interior vertices have the same number of
    channels as the max of the channels of the vertices it feeds into. The output
    channels are divided amongst the vertices that are directly connected to it.
    When the division is not even, some vertices may receive an extra channel to
    compensate.
    Returns:
        list of channel counts, in order of the vertices.
    """
    num_vertices = np.shape(matrix)[0]

    vertex_channels = [0] * num_vertices
    vertex_channels[0] = in_channels
    vertex_channels[num_vertices - 1] = out_channels

    if num_vertices == 2:
        # Edge case where module only has input and output vertices
        return vertex_channels

    # Compute the in-degree ignoring input, axis 0 is the src vertex and axis 1 is
    # the dst vertex. Summing over 0 gives the in-degree count of each vertex.
    in_degree = np.sum(matrix[1:], axis=0)
    interior_channels = out_channels // in_degree[num_vertices - 1]
    correction = out_channels % in_degree[num_vertices - 1]  # Remainder to add

    # Set channels of vertices that flow directly to output
    for v in range(1, num_vertices - 1):
        if matrix[v, num_vertices - 1]:
            vertex_channels[v] = interior_channels
            if correction:
                vertex_channels[v] += 1
                correction -= 1

    # Set channels for all other vertices to the max of the out edges, going
    # backwards. (num_vertices - 2) index skipped because it only connects to
    # output.
    for v in range(num_vertices - 3, 0, -1):
        if not matrix[v, num_vertices - 1]:
            for dst in range(v + 1, num_vertices - 1):
                if matrix[v, dst]:
                    vertex_channels[v] = max(vertex_channels[v], vertex_channels[dst])
        assert vertex_channels[v] > 0

    # Sanity check, verify that channels never increase and final channels add up.
    final_fan_in = 0
    for v in range(1, num_vertices - 1):
        if matrix[v, num_vertices - 1]:
            final_fan_in += vertex_channels[v]
        for dst in range(v + 1, num_vertices - 1):
            if matrix[v, dst]:
                assert vertex_channels[v] >= vertex_channels[dst]
    assert final_fan_in == out_channels or num_vertices == 2
    # num_vertices == 2 means only input/output nodes, so 0 fan-in

    return vertex_channels
