import unittest

from naslib.tabular.networks.mlpnet import *


class MlpNetTest(unittest.TestCase):

    def test_mlp_net(self):
        test_config = {"activation": "relu",
                       "num_layers": 2,
                       "num_units_1": 2,
                       "num_units_2": 2,
                       "use_dropout": True,
                       "dropout_1": 0.5,
                       "dropout_2": 0.0}

        test_mlpnet = MlpNet(test_config, in_features=1, out_features=1, embedding=None, final_activation=None)

        print(test_mlpnet.layers)
        print(test_mlpnet.get_config_space())


if __name__ == "__main__":
    unittest.main()
