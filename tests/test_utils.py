import unittest
import os

from naslib.utils import utils
from fvcore.common.config import CfgNode
class UtilsTest(unittest.TestCase):

    def test_get_config_from_args__from_config_file(self):
        args = utils.parse_args(args=['--config-file', 'assets/config.yaml'])
        config = utils.get_config_from_args(args)

        self.assertEqual(config.seed, 12)
        self.assertEqual(config.search.batch_size, 300)
        self.assertEqual(config.evaluation.batch_size, 200)

    def test_get_config_from_args__from_args(self):
        args = utils.parse_args(args=['seed', '1', 'out_dir', 'tmp/util_test'])
        config = utils.get_config_from_args(args)

        self.assertEqual(config.seed, 1)
        self.assertEqual(config.out_dir, 'tmp/util_test')

    def test_get_config_from_args__default(self):
        self._test_get_config_from_args_default(config_type="predictor")


    ####### Helper methods #######
    def _test_get_config_from_args_default(self, config_type):
        config_filepath = self._get_config_path(config_type=config_type)
        with open(file=config_filepath) as f:
            config_parent = CfgNode.load_cfg(f)

        config_child = utils.get_config_from_args()
        self.assertTrue(self._verify_child_config_consistent(config_parent, config_child))


    def _verify_child_config_consistent(self, parent_config, child_config):
        """ Returns True if all the attributes present in both configs have the same value in both of them.
            child_config may have additional keys, which are ignored.
        """
        for k in parent_config:
            if k not in child_config:
                # print(f"{k} not found in child_config")
                return False

            if isinstance(parent_config[k], CfgNode):
                return self._verify_child_config_consistent(parent_config[k], child_config[k])
            else:
                if parent_config[k] != child_config[k]:
                    # print(f'{parent_config[k]} != {child_config[k]} for {k} in parent_config')
                    return False

        return True

    def _get_config_path(self, config_type):
        config_paths = {
            "nas": "configs/darts_defaults.yaml",
            "predictor": "configs/predictor_config.yaml"
        }

        config_path_full = os.path.join(
            *(
                [utils.get_project_root()] + config_paths[config_type].split('/')
            )
        )

        return config_path_full

class LoggingTest(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
