import unittest

from naslib.tabular.networks.shapedresnet import *


class ShapedResNetTest(unittest.TestCase):

    def test_shaped_res_net(self):
        test_config = {"num_groups":1,
                       "blocks_per_group": 1,
                       "max_units": 101,
                       "activation": "relu",
                       "max_shake_drop_probability": 0.4,
                       "max_dropout": 0.4,
                       "num_layers": 8,
                       "use_dropout": True,
                       "use_shake_shake": True,
                       "use_shake_drop": True}

        shaped_resnet_configspace = ShapedResNet.get_config_space()

        for shape in shaped_resnet_configspace.get_hyperparameter("resnet_shape").choices:
            test_config["resnet_shape"] = shape
            test_config["dropout_shape"] = shape
            test_shaped_resnet = ShapedResNet(test_config, in_features=1, out_features=1, embedding=None)
            print("ShapedResNet with shape", shape, "tested successfully")

        print(test_shaped_resnet.layers)


if __name__ == "__main__":
    unittest.main()
