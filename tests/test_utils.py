import unittest
import os

from naslib import utils
from fvcore.common.config import CfgNode
class UtilsTest(unittest.TestCase):

    def test_get_config_from_args_config_file(self):
        args = utils.parse_args(args=['--config-file', 'assets/config.yaml', '--resume'])
        config = utils.get_config_from_args(args)

        self.assertTrue(args.resume)
        self.assertFalse(args.eval_only)

        self.assertEqual(config.seed, 12)
        self.assertEqual(config.search.batch_size, 300)
        self.assertEqual(config.evaluation.batch_size, 200)

    def test_get_config_from_args_config_args(self):
        args = utils.parse_args(args=['seed', '1', 'search.epochs', '42',
                                      'out_dir', 'tmp/util_test'])
        config = utils.get_config_from_args(args)

        self.assertEqual(config.seed, 1)
        self.assertEqual(config.search.epochs, 42)

    ####### Tests for get_config_from_args with default args #######
    def test_get_config_from_args_default_nas(self):
        self._test_get_config_from_args_default(config_type="nas")

    def test_get_config_from_args_default_predictor(self):
        self._test_get_config_from_args_default(config_type="predictor")

    def test_get_config_from_args_default_nas_predictor(self):
        self._test_get_config_from_args_default(config_type="nas_predictor")

    def test_get_config_from_args_default_bbo_bs(self):
        self._test_get_config_from_args_default(config_type="bbo-bs")

    def test_get_config_from_args_default_oneshot(self):
        self._test_get_config_from_args_default(config_type="oneshot")

    def test_get_config_from_args_default_statistics(self):
        self._test_get_config_from_args_default(config_type="statistics")

    ####### Tests for get_config_from_args with custom args #######
    def test_get_config_from_args_nas(self):
        self._test_get_config_from_args(config_type="nas")

    def test_get_config_from_args_predictor(self):
        self._test_get_config_from_args(config_type="predictor")

    def test_get_config_from_args_nas_predictor(self):
        self._test_get_config_from_args(config_type="nas_predictor")

    def test_get_config_from_args_bbo_bs(self):
        self._test_get_config_from_args(config_type="bbo-bs")

    def test_get_config_from_args_oneshot(self):
        self._test_get_config_from_args(config_type="oneshot")

    def test_get_config_from_args_statistics(self):
        self._test_get_config_from_args(config_type="statistics")

    ####### Helper methods #######
    def _test_get_config_from_args_default(self, config_type):
        config_filepath = self._get_config_path(config_type=config_type)
        with open(file=config_filepath) as f:
            config_parent = CfgNode.load_cfg(f)

        config_child = utils.get_config_from_args(config_type=config_type)
        self.assertTrue(self._verify_child_config_consistent(config_parent, config_child))

    def _test_get_config_from_args(self, config_type):

        args = ['seed', '9001', 'dataset', 'new_dataset']
        args = utils.parse_args(args=args)

        config_file = self._get_config_path(config_type)

        with open(file=config_file) as f:
            config_parent = CfgNode.load_cfg(f)

        config_child = utils.get_config_from_args(args=args, config_type=config_type)

        self.assertEqual(config_child.seed, 9001)
        self.assertEqual(config_child.dataset, 'new_dataset')

        config_child.seed = config_parent.seed
        config_child.dataset = config_parent.dataset

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
            "nas": "defaults/darts_defaults.yaml",
            "predictor": "runners/predictors/predictor_config.yaml",
            "bbo-bs": "runners/bbo/discrete_config.yaml",
            "nas_predictor": "runners/nas_predictors/discrete_config.yaml",
            "oneshot": "runners/nas_predictors/nas_predictor_config.yaml",
            "statistics": "runners/statistics/statistics_config.yaml"
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
