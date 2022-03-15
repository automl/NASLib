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

import torch
import torch.nn as nn
import torch.nn.functional as F

import types

from . import measure
from ..p_utils import get_layer_metric_array, reshape_elements


def fisher_forward_conv2d(self, x):
    x = F.conv2d(
        x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
    )
    # intercept and store the activations after passing through 'hooked' identity op
    self.act = self.dummy(x)
    return self.act


def fisher_forward_linear(self, x):
    x = F.linear(x, self.weight, self.bias)
    self.act = self.dummy(x)
    return self.act


@measure("fisher", bn=True, mode="channel")
def compute_fisher_per_weight(net, inputs, targets, loss_fn, mode, split_data=1):

    device = inputs.device

    if mode == "param":
        raise ValueError("Fisher pruning does not support parameter pruning.")

    net.train()
    all_hooks = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            # variables/op needed for fisher computation
            layer.fisher = None
            layer.act = 0.0
            layer.dummy = nn.Identity()

            # replace forward method of conv/linear
            if isinstance(layer, nn.Conv2d):
                layer.forward = types.MethodType(fisher_forward_conv2d, layer)
            if isinstance(layer, nn.Linear):
                layer.forward = types.MethodType(fisher_forward_linear, layer)

            # function to call during backward pass (hooked on identity op at output of layer)
            def hook_factory(layer):
                def hook(module, grad_input, grad_output):
                    act = layer.act.detach()
                    grad = grad_output[0].detach()
                    if len(act.shape) > 2:
                        g_nk = torch.sum((act * grad), list(range(2, len(act.shape))))
                    else:
                        g_nk = act * grad
                    del_k = g_nk.pow(2).mean(0).mul(0.5)
                    if layer.fisher is None:
                        layer.fisher = del_k
                    else:
                        layer.fisher += del_k
                    del (
                        layer.act
                    )  # without deleting this, a nasty memory leak occurs! related: https://discuss.pytorch.org/t/memory-leak-when-using-forward-hook-and-backward-hook-simultaneously/27555

                return hook

            # register backward hook on identity fcn to compute fisher info
            layer.dummy.register_backward_hook(hook_factory(layer))

    N = inputs.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data

        net.zero_grad()
        outputs = net(inputs[st:en])
        loss = loss_fn(outputs, targets[st:en])
        loss.backward()

    # retrieve fisher info
    def fisher(layer):
        if layer.fisher is not None:
            return torch.abs(layer.fisher.detach())
        else:
            return torch.zeros(layer.weight.shape[0])  # size=ch

    grads_abs_ch = get_layer_metric_array(net, fisher, mode)

    # broadcast channel value here to all parameters in that channel
    # to be compatible with stuff downstream (which expects per-parameter metrics)
    # TODO cleanup on the selectors/apply_prune_mask side (?)
    shapes = get_layer_metric_array(net, lambda l: l.weight.shape[1:], mode)

    grads_abs = reshape_elements(grads_abs_ch, shapes, device)

    return grads_abs
