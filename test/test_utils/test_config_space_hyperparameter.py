import unittest

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from naslib.utils.config_space_hyperparameter import *


class ConfigSpaceHyperparameterTest(unittest.TestCase):

    def test_get_hyperparameter(self):
        test_hyperparameter_type_1 = CSH.Constant
        test_name_1 = "test_name_1"
        value_range_1 = (1.,1.)

        test_hyperparameter_type_2 = CSH.UniformFloatHyperparameter
        test_name_2 = "test_name_2"
        value_range_2 = ((0.01,1),True)

        r1 = get_hyperparameter(test_hyperparameter_type_1, test_name_1, value_range_1)
        r2 = get_hyperparameter(test_hyperparameter_type_2, test_name_2, value_range_2)

        print(r1)
        print(r2)

    def test_add_hyperparameter(self):
        test_CS = CS.ConfigurationSpace()
        test_hyperparameter_type_1 = CSH.Constant
        test_name_1 = "test_name_1"
        value_range_1 = (1.,1.)

        r_CS = add_hyperparameter(test_CS, test_hyperparameter_type_1, test_name_1, value_range_1)

        print(r_CS)


if __name__ == "__main__":
    unittest.main()
