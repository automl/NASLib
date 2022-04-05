""" Evaluates a ZeroCost predictor across all search spaces and datasets/tasks"""
import argparse
from naslib.utils import utils, setup_logger, get_dataset_api, utils_predictor
from fvcore.common.config import CfgNode

parser = argparse.ArgumentParser()
parser.add_argument('--predictor', default='synflow', type=str, help='predictor')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--test_size', default=5, type=int, help='test_size')
parser.add_argument('--train_size', default=5, type=int, help='train_size')
parser.add_argument('--out_dir', default='run/', type=str, help='out dir')
parser.add_argument('--seed', default=1000, type=int, help='seed')
parser.add_argument('--cutout', default=False, type=bool, help='cutout')
parser.add_argument('--cutout_length', default=16, type=int, help='cutout')
parser.add_argument('--cutout_prob', default=1.0, type=float, help='cutout')
parser.add_argument('--train_portion', default=0.7, type=float, help='train_portion')
config = parser.parse_args()
config=CfgNode(vars(config))
# Test predictor over all search spaces and datasets
utils_predictor.evaluate_predictor_across_search_spaces(config)
