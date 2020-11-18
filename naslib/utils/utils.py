from __future__ import print_function

import sys
import logging
import argparse
import torchvision.datasets as dset

from copy import copy

import random
import os
import os.path
import shutil
from functools import wraps, partial
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
import yaml

from fvcore.common.checkpoint import Checkpointer as fvCheckpointer
from fvcore.common.config import CfgNode

cat_channels = partial(torch.cat, dim=1)

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """
    Returns the root path of the project.
    """
    return Path(__file__).parent.parent


def iter_flatten(iterable):
    """
    Flatten a potentially deeply nested python list
    """
    # taken from https://rightfootin.blogspot.com/2006/09/more-on-python-flatten.html
    it = iter(iterable)
    for e in it:
        if isinstance(e, (list, tuple)):
            for f in iter_flatten(e):
                yield f
        else:
            yield e


def default_argument_parser():
    """
    Returns the argument parser with the default options.

    Inspired by the implementation of FAIR's detectron2
    """

    parser = argparse.ArgumentParser(
        epilog=f"""
Examples:
Run on single machine:
    $ {sys.argv[0]} --config-file cfg.yaml dataset 'cifar100' seed 1
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="{}/defaults/darts_defaults.yaml".format(get_project_root()), metavar="FILE", help="path to config file")
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def parse_args(parser=default_argument_parser(), args=sys.argv[1:]):
    return parser.parse_args(args)


def pairwise(iterable):
    """
    Iterate pairwise over list.

    from https://stackoverflow.com/questions/5389507/iterating-over-every-two-elements-in-a-list
    """
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)


def get_config_from_args(args=None):
    """
    Parses command line arguments and merges them with the defaults
    from the config file.

    Prepares experiment directories.

    Args:
        args: args from a different argument parser than the default one.
    """
    # load the default base
    with open(os.path.join(get_project_root(), 'defaults', 'darts_defaults.yaml')) as f:
        config = CfgNode.load_cfg(f)
    
    if not args:
        args = parse_args()
    logger.info("Command line args: {}".format(args))
    
    config.eval_only = args.eval_only
    config.resume = args.resume
    
    # load config file
    config.merge_from_file(args.config_file)
    config.merge_from_list(args.opts)

    # prepare the output directories
    config.save = '{}/{}/{}/{}'.format(config.out_dir, config.dataset, config.optimizer, config.seed)
    config.data = "{}/data".format(get_project_root())

    create_exp_dir(config.save)
    create_exp_dir(config.save + "/search")     # required for the checkpoints
    create_exp_dir(config.save + "/eval")

    return config


def get_train_val_loaders(config, mode):
    """
    Constructs the dataloaders and transforms for training, validation and test data.
    """
    data = config.data
    dataset = config.dataset
    seed = config.seed
    config = config.search if mode=='train' else config.evaluation
    if dataset == 'cifar10':
        train_transform, valid_transform = _data_transforms_cifar10(config)
        train_data = dset.CIFAR10(root=data, train=True, download=True, transform=train_transform)
        test_data = dset.CIFAR10(root=data, train=False, download=True, transform=valid_transform)
    elif dataset == 'cifar100':
        train_transform, valid_transform = _data_transforms_cifar100(config)
        train_data = dset.CIFAR100(root=data, train=True, download=True, transform=train_transform)
        test_data = dset.CIFAR100(root=data, train=False, download=True, transform=valid_transform)
    elif dataset == 'svhn':
        train_transform, valid_transform = _data_transforms_svhn(config)
        train_data = dset.SVHN(root=data, split='train', download=True, transform=train_transform)
        test_data = dset.SVHN(root=data, split='test', download=True, transform=valid_transform)
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(config.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=0, worker_init_fn=np.random.seed(seed))

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=0, worker_init_fn=np.random.seed(seed))

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=config.batch_size, shuffle=False,
        pin_memory=True, num_workers=0, worker_init_fn=np.random.seed(seed))

    return train_queue, valid_queue, test_queue, train_transform, valid_transform


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length,
                                                 args.cutout_prob))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def _data_transforms_svhn(args):
    SVHN_MEAN = [0.4377, 0.4438, 0.4728]
    SVHN_STD = [0.1980, 0.2010, 0.1970]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(SVHN_MEAN, SVHN_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length,
                                                 args.cutout_prob))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(SVHN_MEAN, SVHN_STD),
    ])
    return train_transform, valid_transform


def _data_transforms_cifar100(args):
    CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
    CIFAR_STD = [0.2673, 0.2564, 0.2762]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length,
                                                 args.cutout_prob))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def set_seed(seed):
    """
    Set the seeds for all used libraries.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)


def get_last_checkpoint(config, search=True):
    """
    Finds the latest checkpoint in the experiment directory.

    Args:
        config (AttrDict): The config from config file.
        search (bool): Search or evaluation checkpoint
    
    Returns:
        (str): The path to the latest checkpoint file.
    """
    try:
        path = os.path.join(config.save, "search" if search else "eval", "last_checkpoint")
        with open(path, 'r') as f:
            checkpoint_name = f.readline()
        return os.path.join(config.save, "search" if search else "eval", checkpoint_name)
    except:
        return ""


def accuracy(output, target, topk=(1,)):
    """
    Calculate the accuracy given the softmax output and the target.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def count_parameters_in_MB(model):
    """
    Returns the model parameters in mega byte.
    """
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if
                  "auxiliary" not in name) / 1e6


def log_args(args):
    """
    Log the args in a nice way.
    """
    for arg, val in args.items():
        logger.info(arg + '.' * (50 - len(arg) - len(str(val))) + str(val))


def create_exp_dir(path):
    """
    Create the experiment directories.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    logger.info('Experiment dir : {}'.format(path))


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt



class Cutout(object):
    def __init__(self, length, prob=1.0):
        self.length = length
        self.prob = prob

    def __call__(self, img):
        if np.random.binomial(1, self.prob):
            h, w = img.size(1), img.size(2)
            mask = np.ones((h, w), np.float32)
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.
            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img *= mask
        return img


from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple
from fvcore.common.file_io import PathManager
import os

class Checkpointer(fvCheckpointer):


    def load(self, path: str, checkpointables: Optional[List[str]] = None) -> object:
        """
        Load from the given checkpoint. When path points to network file, this
        function has to be called on all ranks.
        Args:
            path (str): path or url to the checkpoint. If empty, will not load
                anything.
            checkpointables (list): List of checkpointable names to load. If not
                specified (None), will load all the possible checkpointables.
        Returns:
            dict:
                extra data loaded from the checkpoint that has not been
                processed. For example, those saved with
                :meth:`.save(**extra_data)`.
        """
        if not path:
            # no checkpoint provided
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(path))
        if not os.path.isfile(path):
            path = PathManager.get_local_path(path)
            assert os.path.isfile(path), "Checkpoint {} not found!".format(path)

        checkpoint = self._load_file(path)
        incompatible = self._load_model(checkpoint)
        if (
            incompatible is not None
        ):  # handle some existing subclasses that returns None
            self._log_incompatible_keys(incompatible)

        for key in self.checkpointables if checkpointables is None else checkpointables:
            if key in checkpoint:  # pyre-ignore
                self.logger.info("Loading {} from {}".format(key, path))
                obj = self.checkpointables[key]
                try:
                    obj.load_state_dict(checkpoint.pop(key))  # pyre-ignore
                except:
                    print("exception loading")
        
        # return any further checkpoint data
        return checkpoint