#Copyright (C) 2010-2021 Alibaba Group Holding Limited.
# =============================================================================

import torch

from torch import nn
from . import measure


@measure("zen", bn=True)
def compute_zen_score(net, inputs, targets, loss_fn=None, split_data=1,
                      repeat=1, mixup_gamma=1e-2, fp16=False):
    nas_score_list = []

    device = inputs.device
    dtype = torch.half if fp16 else torch.float32

    from IPython import embed
    embed()

    with torch.no_grad():
        input = torch.randn(size=list(inputs.shape), device=device, dtype=dtype)
        input2 = torch.randn(size=list(inputs.shape), device=device, dtype=dtype)
        mixup_input = input + mixup_gamma * input2
        output = net.forward(input)
        mixup_output = net.forward(mixup_input)

        nas_score = torch.sum(torch.abs(output - mixup_output), dim=[1, 2, 3])
        nas_score = torch.mean(nas_score)

        # compute BN scaling
        log_bn_scaling_factor = 0.0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                bn_scaling_factor = torch.sqrt(torch.mean(m.running_var))
                log_bn_scaling_factor += torch.log(bn_scaling_factor)
            pass
        pass
        nas_score = torch.log(nas_score) + log_bn_scaling_factor
        nas_score_list.append(float(nas_score))

    avg_nas_score = float(np.mean(nas_score_list))

    return avg_nas_score
