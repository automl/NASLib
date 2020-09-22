import torch

from naslib.search_spaces.core.primitives import AbstractPrimitive, Identity

class DropPathWrapper(AbstractPrimitive):
    """
    A wrapper for the drop path training regularization.
    """

    def __init__(self, op):
        super().__init__(locals())
        self.op = op


    def forward(self, x, edge_data):
        x = self.op(x, edge_data)
        if edge_data.drop_path_prob > 0. and not isinstance(self.op, Identity) and self.training:
            keep_prob = 1. - edge_data.drop_path_prob
            mask = torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
            if torch.cuda.is_available():
                mask = mask.cuda()
            x.div_(keep_prob)
            x.mul_(mask)
        return x


    def get_embedded_ops(self):
        return self.op