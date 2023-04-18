import unittest

import torch.nn as nn
import torch.optim as optim

from naslib.utils.pytorch_helper import create_optimizer


class OptimizerFactoryTest(unittest.TestCase):
    def setUp(self):
        self.model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))

    def test_params_type_correct(self):
        create_optimizer("sgd", self.model.parameters())

    def test_params_type_false(self):
        with self.assertRaises(ValueError):
            create_optimizer("sgd", self.model)

    def test_sgd(self):
        self._test_optimizer_helper('sgd', optim.SGD, lr=0.026, weight_decay=0, momentum=0.91)

    def test_adam(self):
        self._test_optimizer_helper('Adam', optim.Adam, lr=0.026, weight_decay=0, betas=(0.5, 0.999))

    # Helper functions
    def _test_optimizer_helper(self, opt, optimizer_type, lr, weight_decay, **kwargs):
        optimizer = create_optimizer(
            opt,
            self.model.parameters(),
            lr,
            weight_decay,
            **kwargs
        )

        self.assertIsInstance(optimizer, optimizer_type)
        self.assertEqual(optimizer.defaults["lr"], lr)
        self.assertEqual(optimizer.defaults["weight_decay"], weight_decay)
        for item in kwargs:
            self.assertEqual(optimizer.defaults[item], kwargs[item])


if __name__ == '__main__':
    unittest.main()
