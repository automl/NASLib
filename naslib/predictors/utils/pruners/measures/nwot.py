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
"""
This contains implementations of nwot based on the updated version of
https://github.com/BayesWatch/nas-without-training
to reflect the second version of the paper https://arxiv.org/abs/2006.04647
"""

import torch
import numpy as np

from . import measure


@measure("nwot", bn=True)
def compute_nwot(net, inputs, targets, split_data=1, loss_fn=None):
    batch_size = len(targets)

    def counting_forward_hook(module, inp, out):
        inp = inp[0].view(inp[0].size(0), -1)
        x = (inp > 0).float()  # binary indicator
        K = x @ x.t()
        K2 = (1. - x) @ (1. - x.t())
        net.K = net.K + K.cpu().numpy() + K2.cpu().numpy()  # hamming distance

    def counting_backward_hook(module, inp, out):
        module.visited_backwards = True

    net.K = np.zeros((batch_size, batch_size))
    for name, module in net.named_modules():
        module_type = str(type(module))
        if ('ReLU' in module_type) and ('naslib' not in module_type):
            # module.register_full_backward_hook(counting_backward_hook)
            module.register_forward_hook(counting_forward_hook)

    x = torch.clone(inputs)
    net(x)
    s, jc = np.linalg.slogdet(net.K)

    return jc