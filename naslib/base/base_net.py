# -*- coding: utf-8 -*-

"""
Parent Class of all Networks based on features.
"""

import torch.nn as nn
from collections import OrderedDict
import ConfigSpace

__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

class BaseNet(nn.Module):
    """ Parent class for all Networks"""
    def __init__(self, config, in_features, out_features, final_activation):
        """
        Initialize the BaseNet.
        """

        super(BaseNet, self).__init__()
        self.layers = nn.Sequential()
        self.config = config
        self.n_feats = in_features
        self.n_classes = out_features
        self.epochs_trained = 0
        self.budget_trained = 0
        self.stopped_early = False
        self.last_compute_result = None
        self.logs = []
        self.num_epochs_no_progress = 0
        self.current_best_epoch_performance = None
        self.best_parameters = None
        self.final_activation = final_activation

    def forward(self, x):
        x = self.layers(x)
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)
        return x

    def snapshot(self):
        self.best_parameters = OrderedDict({key: value.cpu().clone() for key, value in self.state_dict().items()})

    def load_snapshot(self):
        if self.best_parameters is not None:
            self.load_state_dict(self.best_parameters)

    def set_net_from_hyperpar_config(self, hyperpar_config, in_features, out_features):
        """
        Set layers according to a hyperparameter config as returned by Auto PyTorch .fit method.
        """
        from ..utils.configspace_wrapper import ConfigWrapper

        config = ConfigWrapper("NetworkSelector", hyperpar_config)

        networks = self._get_default_network_dict()

        network_type = networks[config["network"]]
        network_config = ConfigWrapper(config["network"], config)

        network = network_type(config=network_config,
                               in_features=in_features, out_features=out_features,
                               embedding=nn.Sequential(), final_activation=None)

        self.layers = network.layers

    def _get_default_network_dict(self):

        from naslib import MlpNet, ResNet, ShapedMlpNet, ShapedResNet

        networks = dict()
        networks["mlpnet"] = MlpNet
        networks["resnet"] = ResNet
        networks["shapedmlpnet"] = ShapedMlpNet
        networks["shapedresnet"] = ShapedResNet

        return networks

    @staticmethod
    def get_config_space():
        return ConfigSpace.ConfigurationSpace()


class BaseFeatureNet(BaseNet):
    """ Parent class for MlpNet, ResNet, ... Can use entity embedding for cagtegorical features"""
    def __init__(self, config, in_features, out_features, embedding, final_activation):
        """
        Initialize the BaseFeatureNet.
        """

        super(BaseFeatureNet, self).__init__(config, in_features, out_features, final_activation)
        self.embedding = embedding

    def forward(self, x):
        x = self.embedding(x)
        return super(BaseFeatureNet, self).forward(x)
