import unittest

from naslib.base.base_net import *


class BaseNetTest(unittest.TestCase):

    def test_base_net(self):
        test_config = {"NetworkSelector:network":"shapedresnet",
                       "NetworkSelector:shapedresnet:activation": "relu",
                       "NetworkSelector:shapedresnet:blocks_per_group": 4,
                       "NetworkSelector:shapedresnet:max_units": 172,
                       "NetworkSelector:shapedresnet:num_groups": 1,
                       "NetworkSelector:shapedresnet:resnet_shape": "stairs",
                       "NetworkSelector:shapedresnet:use_dropout": False,
                       "NetworkSelector:shapedresnet:use_shake_drop": False,
                       "NetworkSelector:shapedresnet:use_shake_shake": True}

        base_net = BaseNet(test_config, in_features = 1, out_features=1, final_activation=None)

        base_net.forward(1)
        base_net.snapshot()
        base_net.load_snapshot()
        base_net.set_net_from_hyperpar_config(test_config, in_features=1, out_features=1)
        print(base_net.layers)
        print(base_net._get_default_network_dict())
        print(base_net.get_config_space())


if __name__ == "__main__":
    unittest.main()
