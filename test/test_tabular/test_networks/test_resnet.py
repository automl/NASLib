import unittest

from naslib.tabular.networks.resnet import *


class ResNetTest(unittest.TestCase):

    def test_resnet(self):
        test_config = {"activation": "relu",
                       "num_groups": 2,
                       "blocks_per_group": 1,
                       "num_units_0": 10,
                       "num_units_1": 10,
                       "num_units_2": 10,
                       "max_shake_drop_probability": 0.5,
                       "dropout": 0.5,
                       "use_shake_drop": True,
                       "use_shake_shake": True,
                       "use_dropout": True,
                       "dropout_1": 0.5,
                       "dropout_2": 0.5}

        test_resnet = ResNet(test_config, in_features=1, out_features=1, embedding=None, final_activation=None)

        print(test_resnet.layers)
        print(test_resnet.get_config_space())


if __name__ == "__main__":
    unittest.main()
