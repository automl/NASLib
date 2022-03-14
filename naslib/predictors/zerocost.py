"""
This contains implementations of:
synflow, grad_norm, fisher, and grasp, and variants of jacov and snip
based on https://github.com/mohsaied/zero-cost-nas
"""
import torch
import logging
import math

from naslib.predictors.predictor import Predictor
from naslib.utils.utils import get_project_root, get_train_val_loaders
from naslib.predictors.utils.pruners import predictive

logger = logging.getLogger(__name__)


class ZeroCost(Predictor):
    def __init__(self, config, batch_size=64, method_type="jacov"):
        # available zero-cost method types: 'jacov', 'snip', 'synflow', 'grad_norm', 'fisher', 'grasp'
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.batch_size = batch_size
        self.method_type = method_type
        self.dataload = "random"
        self.num_imgs_or_batches = 1
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        config.data = "{}/data".format(get_project_root())
        self.config = config

    def pre_process(self):
        self.train_loader, _, _, _, _ = \
                get_train_val_loaders(self.config, mode="train")

    def query(self, xtest, info=None):
        graph = xtest
        loss_fn = graph.get_loss_fn()

        n_classes = graph.num_classes
        score = predictive.find_measures(
                net_orig=graph,
                dataloader=self.train_loader,
                dataload_info=(self.dataload, self.num_imgs_or_batches, n_classes),
                device=self.device,
                loss_fn=loss_fn,
                measure_names=[self.method_type],
            )

        if math.isnan(score):
            score = -1e8

        return score
