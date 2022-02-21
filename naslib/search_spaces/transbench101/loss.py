import torch
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss

class SoftmaxCrossEntropyWithLogits(_WeightedLoss):
    def __init__(self, weight=None):
        super(SoftmaxCrossEntropyWithLogits, self).__init__(weight=None)
        self.weight = weight

    def forward(self, input, target):
        logits_scaled = torch.log(F.softmax(input, dim=-1) + 0.00001)

        if self.weight is not None:
            loss = -((target * logits_scaled) * self.weight).sum(dim=-1)
        else:
            loss = -(target * logits_scaled).sum(dim=-1)
        return loss.mean()
