import unittest

from naslib.tabular.networks.shapedmlpnet import *


class ShapedMlpNetTest(unittest.TestCase):

    def test_shaped_mlp_net(self):
        test_config = {"activation": "relu",
                       "max_dropout": 0.4,
                       "max_units": 101,
                       "num_layers": 8,
                       "use_dropout": True}

        shaped_mlpnet_configspace = ShapedMlpNet.get_config_space()

        for shape in shaped_mlpnet_configspace.get_hyperparameter("mlp_shape").choices:
            test_config["mlp_shape"] = shape
            test_config["dropout_shape"] = shape
            ShapedMlpNet(test_config, in_features=1, out_features=1, embedding=None)
            print("ShapedMlpNet with shape", shape, "tested successfully")


if __name__ == "__main__":
    unittest.main()
