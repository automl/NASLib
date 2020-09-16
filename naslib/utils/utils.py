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

import numpy as np
import torch
import torchvision.transforms as transforms
import yaml
from torch.autograd import Variable

cat_channels = partial(torch.cat, dim=1)

logger = logging.getLogger(__name__)

def iter_flatten(iterable):
    # taken from https://rightfootin.blogspot.com/2006/09/more-on-python-flatten.html
    it = iter(iterable)
    for e in it:
        if isinstance(e, (list, tuple)):
            for f in iter_flatten(e):
                yield f
        else:
            yield e


def exception(exception_type):
    def exception_decorator(func):
        @wraps(func)
        def function_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception_type:
                return None

        return function_wrapper

    return exception_decorator


# Inspired by the implementation of FAIR's detectron2
def default_argument_parser():
    """
    Returns the argument parser with the available options
    """

    parser = argparse.ArgumentParser(
        epilog=f"""
Examples:
Run on single machine:
    $ {sys.argv[0]} --config-file cfg.yaml dataset 'cifar100' seed 1
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="../../configs/default.yaml", metavar="FILE", help="path to config file")
    parser.add_argument("--seed", default=1, type=int, help="Seed for the experiment")
    parser.add_argument("--optimizer", default="na")
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


# from https://stackoverflow.com/questions/5389507/iterating-over-every-two-elements-in-a-list
def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)


def get_config_from_args(args=None):
    """
    Parses command line arguments and merges them with the defaults
    from the config file.

    Prepares experiment directories.
    """
    if not args:
        args = default_argument_parser().parse_args()
    logger.info("Command line args: {}".format(args))
    
    # load config file
    with open(args.config_file, 'r') as f:
        config = AttrDict(yaml.safe_load(f))
    for k, v in config.items():
        if isinstance(v, dict):
            config[k] = AttrDict(v)

    # Override file args with ones from command line
    for arg, value in pairwise(args.opts):
        config[arg] = value

    config.optimizer = args.optimizer
    config.eval_only = args.eval_only
    config.seed = args.seed
    config.search.seed = config.evaluation.seed = config.seed

    config._save = copy(config.save)
    config.save = '{}/{}/{}/{}'.format(config.save, config.dataset, config.optimizer, config.seed)

    create_exp_dir(config.save)

    if config.dataset != 'cifar100':
        config.n_classes = 10
    else:
        config.n_classes = 100
    return config


def get_train_val_loaders(config):
    """
    Constructs the dataloaders and transforms for training, validation and test data.
    """
    if config.dataset == 'cifar10':
        train_transform, valid_transform = _data_transforms_cifar10(config)
        train_data = dset.CIFAR10(root=config.data, train=True, download=True, transform=train_transform)
        test_data = dset.CIFAR10(root=config.data, train=False, download=True, transform=valid_transform)
    elif config.dataset == 'cifar100':
        train_transform, valid_transform = _data_transforms_cifar100(config)
        train_data = dset.CIFAR100(root=config.data, train=True, download=True, transform=train_transform)
        test_data = dset.CIFAR100(root=config.data, train=False, download=True, transform=valid_transform)
    elif config.dataset == 'svhn':
        train_transform, valid_transform = _data_transforms_svhn(config)
        train_data = dset.SVHN(root=config.data, split='train', download=True, transform=train_transform)
        test_data = dset.SVHN(root=config.data, split='test', download=True, transform=valid_transform)
    else:
        raise ValueError("Unknown dataset: {}".format(config.dataset))

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(config.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=0, worker_init_fn=np.random.seed(config.seed))

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=0, worker_init_fn=np.random.seed(config.seed))

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=config.batch_size, shuffle=False,
        pin_memory=True, num_workers=0, worker_init_fn=np.random.seed(config.seed))

    return train_queue, valid_queue, test_queue, train_transform, valid_transform


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)


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


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


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


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if
                  "auxiliary" not in name) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def log_args(args):
    for arg, val in args.items():
        logger.info(arg + '.' * (50 - len(arg) - len(str(val))) + str(val))


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


def create_exp_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    logger.info('Experiment dir : {}'.format(path))

