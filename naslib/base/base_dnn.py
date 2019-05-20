import torch.nn as nn

class BaseDNN(nn.Module):
    """
    Parent class for dnns.
    """
    def __init__(self):
        super(BaseDNN, self).__init__()
        self.layers = nn.Sequential()
        self.networks = self._get_default_network_dict()

    def forward(self, x):
        return self.layers(x)
        
    def set_net_from_hyperpar_config(self, hyperpar_config, in_features, out_features):
        """
        Set layers according to a hyperparameter config as returned by Auto PyTorch .fit method.
        """
        from autoPyTorch.utils.configspace_wrapper import ConfigWrapper
        from autoPyTorch.pipeline.nodes.network_selector import NetworkSelector
        
        config = ConfigWrapper(NetworkSelector().get_name(), hyperpar_config)
        
        network_type = self.networks[config["network"]]
        network_config = ConfigWrapper(config["network"], config)
        
        network = network_type(config=network_config, 
                               in_features=in_features, out_features=out_features,
                               embedding=nn.Sequential(), final_activation=None)
        
        self.layers = network.layers
        
    def _get_default_network_dict(self):
        
        from autoPyTorch.components.networks.feature import MlpNet, ResNet, ShapedMlpNet, ShapedResNet
        
        networks = dict()
        networks["mlpnet"] = MlpNet
        networks["resnet"] = ResNet
        networks["shapedmlpnet"] = ShapedMlpNet
        networks["shapedresnet"] = ShapedResNet
        
        return networks
