import unittest

from naslib.utils.configspace_wrapper import *


class ConfigSpaceWrapperTest(unittest.TestCase):

    def test_configspace_wrapper(self):
        test_config = {"CreateDataLoader:batch_size": 141,
                       "NetworkSelector:shapedresnet:activation": "relu",
                       "NetworkSelector:shapedresnet:blocks_per_group": 4,
                       "NetworkSelector:shapedresnet:max_units": 172,
                       "NetworkSelector:shapedresnet:num_groups": 1,
                       "NetworkSelector:shapedresnet:resnet_shape": "stairs",
                       "NetworkSelector:shapedresnet:use_dropout": False,
                       "NetworkSelector:shapedresnet:use_shake_drop": False,
                       "OptimizerSelector:adam:learning_rate": 0.002512532708432675}

        test_wrapper = ConfigWrapper("NetworkSelector", test_config)
        test_wrapper.update({"NetworkSelector:shapedresnet:use_shake_shake": True})
        print("Test dictionary:", test_wrapper.get_dictionary())
        print("Test indexing:", test_wrapper["shapedresnet:activation"])
        print("Tets contain method:", "shapedresnet:activation" in test_wrapper)
        

if __name__ == "__main__":
    unittest.main()
