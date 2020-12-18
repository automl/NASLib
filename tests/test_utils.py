import unittest
import logging
import torch
import os
import argparse

from collections import namedtuple

from naslib.search_spaces import SimpleCellSearchSpace
from naslib.optimizers import DARTSOptimizer, GDASOptimizer
from naslib.utils import utils, setup_logger


class UtilsTest(unittest.TestCase):

    def test_get_config_from_args_config_file(self):
        args = utils.parse_args(args=['--config-file', 'tests/assets/config.yaml', '--resume'])
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


class LoggingTest(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
