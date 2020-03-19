import os
import sys
import time
import argparse
import logging
import json
import numpy as np

from naslib.optimizers.oneshot import DARTS
from naslib.search_spaces.nasbench1shot1.search_spaces import SearchSpace1
from naslib.utils.utils import config_parser


def main(_args):
    #TODO: move this to the searchers
    args = config_parser(
        '/home/zelaa/NASLib/naslib/configs/default.yaml'
    )
    args.__dict__.update(_args.__dict__)

    save_dir = args.save
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if args.eval_only:
        assert args.save_dir is not None

    # Dump the config of the run folder
    with open(os.path.join(save_dir, 'config.json'), 'w') as fp:
        json.dump(args.__dict__, fp)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info(args)

    search_space = eval('SearchSpace{}()'.format(args.search_space))
    searcher = DARTS(args, search_space)

    if not args.eval_only:
        searcher.run()
        archs = searcher.get_eval_arch()
    else:
        np.random.seed(args.seed + 1)
        archs = searcher.get_eval_arch(2)

    logging.info(archs)
    arch = ' '.join([str(a) for a in archs[0][0]])
    with open('/tmp/arch', 'w') as f:
        f.write(arch)

    return arch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args for SHA with weight sharing')
    parser.add_argument('--seed', dest='seed', type=int, default=100)
    parser.add_argument('--epochs', dest='epochs', type=int, default=50)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
    parser.add_argument('--grad_clip', dest='grad_clip', type=float, default=0.25)
    parser.add_argument('--save_dir', dest='save_dir', type=str, default=None)
    parser.add_argument('--eval_only', dest='eval_only', type=int, default=0)
    # CIFAR-10 only argument.  Use either 16 or 24 for the settings for random_ws search
    # with weight-sharing used in our experiments.
    parser.add_argument('--init_channels', dest='init_channels', type=int, default=16)
    parser.add_argument('--search_space', choices=['1', '2', '3'], default='1')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--data_size', dest='data_size', type=int, default=25000)
    parser.add_argument('--time_steps', dest='time_steps', type=int, default=1)
    _args = parser.parse_args()

    main(_args)

