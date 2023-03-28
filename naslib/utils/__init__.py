from __future__ import print_function

import os
import sys
import logging
import argparse
import torchvision.datasets as dset
from torch.utils.data import Dataset
from sklearn import metrics
from scipy import stats
import copy

from collections import OrderedDict

import random
import os.path
from functools import partial
from pathlib import Path
from typing import List, Optional
from naslib.utils.log import setup_logger

from fvcore.common.checkpoint import Checkpointer as fvCheckpointer
from fvcore.common.config import CfgNode
from fvcore.common.file_io import PathManager

import numpy as np
import torch
import torchvision.transforms as transforms
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

# Expose apis
from .taskonomy_dataset import get_datasets

from .get_dataset_api import get_dataset_api, get_zc_benchmark_api, load_sampled_architectures
from .dataset import get_train_val_loaders
from .zerocost import p_at_tb_k, minmax_n_at_k, compute_scores

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
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default=None,
                        metavar="FILE", help="path to config file")
    parser.add_argument(
        "--eval-only", action="store_true", help="perform evaluation only"
    )
    parser.add_argument("--seed", default=None, help="random seed")
    parser.add_argument(
        "--resume", action="store_true", help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--model-path", type=str, default=None, help="Path to saved model weights"
    )
    parser.add_argument(
        "--world-size",
        default=1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="node rank for distributed training"
    )
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:8888",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--multiprocessing-distributed",
        action="store_true",
        help="Use multi-processing distributed training to launch "
             "N processes per node, which has N GPUs. This is the "
             "fastest way to use PyTorch for either single node or "
             "multi node data parallel training",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--datapath", default=None, metavar="FILE",
                        help="Path to the folder with train/test data folders")
    return parser


def parse_args(parser=default_argument_parser(), args=sys.argv[1:]):
    if "-f" in args:
        args = args[2:]
    return parser.parse_args(args)


def pairwise(iterable):
    """
    Iterate pairwise over list.

    from https://stackoverflow.com/questions/5389507/iterating-over-every-two-elements-in-a-list
    """
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)


def load_config(path):
    with open(path) as f:
        config = CfgNode.load_cfg(f)

    return config


def load_default_config(config_type="nas"):
    config_paths = {
        "nas": "defaults/darts_defaults.yaml",
        "predictor": "runners/predictors/predictor_config.yaml",
        "bbo-bs": "runners/bbo/discrete_config.yaml",
        "nas_predictor": "runners/nas_predictors/discrete_config.yaml",
        "oneshot": "runners/nas_predictors/nas_predictor_config.yaml",
        "statistics": "runners/statistics/statistics_config.yaml",
        "zc": "runners/zc/zc_config.yaml",
    }

    config_path_full = os.path.join(
        *(
            [get_project_root()] + config_paths[config_type].split('/')
        )
    )

    return load_config(config_path_full)


def create_exp_dir(path):
    """
    Create the experiment directories.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    logger.info("Experiment dir : {}".format(path))


def get_config_from_args(args=None, config_type="nas"):
    """
    Parses command line arguments and merges them with the defaults
    from the config file.

    Prepares experiment directories.

    Args:
        args: args from a different argument parser than the default one.
    """

    if args is None:
        args = parse_args()
    logger.info("Command line args: {}".format(args))

    if args.config_file is None:
        config = load_default_config(config_type=config_type)
    else:
        config = load_config(path=args.config_file)

    # Override file args with ones from command line
    try:
        for arg, value in pairwise(args.opts):
            if "." in arg:
                arg1, arg2 = arg.split(".")
                config[arg1][arg2] = type(config[arg1][arg2])(value)
            else:
                config[arg] = type(config[arg])(
                    value) if arg in config else eval(value)

        config.eval_only = args.eval_only
        config.resume = args.resume
        config.model_path = args.model_path

        # load config file
        config.set_new_allowed(True)
        config.merge_from_list(args.opts)

    except AttributeError:
        for arg, value in pairwise(args):
            config[arg] = value

    # prepare the output directories
    if config_type == "nas":
        config.search.seed = config.seed
        config.evaluation.world_size = args.world_size
        config.gpu = config.search.gpu = config.evaluation.gpu = args.gpu
        config.evaluation.rank = args.rank
        config.evaluation.dist_url = args.dist_url
        config.evaluation.dist_backend = args.dist_backend
        config.evaluation.multiprocessing_distributed = args.multiprocessing_distributed
        config.save = "{}/{}/{}/{}/{}".format(
            config.out_dir, config.search_space, config.dataset, config.optimizer, config.seed
        )

    elif config_type == "bbo-bs":
        if not hasattr(config, 'evaluation'):
            config.evaluation = CfgNode()
        config.search.seed = config.seed
        config.evaluation.world_size = args.world_size
        config.gpu = config.search.gpu = config.evaluation.gpu = args.gpu
        config.evaluation.rank = args.rank
        config.evaluation.dist_url = args.dist_url
        config.evaluation.dist_backend = args.dist_backend
        config.evaluation.multiprocessing_distributed = args.multiprocessing_distributed

        if not hasattr(config, "config_id"):  # FIXME
            config.config_id = 0
        config.save = "{}/{}/{}/{}/config_{}/{}".format(
            config.out_dir, config.search_space, config.dataset, config.optimizer, config.config_id, config.seed
        )

    elif config_type == "predictor":
        config.search.seed = config.seed
        if config.predictor == "lcsvr" and config.experiment_type == "vary_train_size":
            config.save = "{}/{}/{}/{}_train/{}".format(
                config.out_dir,
                config.dataset,
                "predictors",
                config.predictor,
                config.seed,
            )
        elif config.predictor == "lcsvr" and config.experiment_type == "vary_fidelity":
            config.save = "{}/{}/{}/{}_fidelity/{}".format(
                config.out_dir,
                config.dataset,
                "predictors",
                config.predictor,
                config.seed,
            )
        else:
            config.save = "{}/{}/{}/{}/{}".format(
                config.out_dir,
                config.dataset,
                "predictors",
                config.predictor,
                config.seed,
            )
    elif config_type == "nas_predictor":
        config.search.seed = config.seed
        config.save = "{}/{}/{}/{}/{}/{}".format(
            config.out_dir,
            config.dataset,
            "nas_predictors",
            config.search_space,
            config.search.predictor_type,
            config.seed,
        )
    elif config_type == "oneshot":
        config.save = "{}/{}/{}/{}/{}/{}".format(
            config.out_dir,
            config.dataset,
            "nas_predictors",
            config.search_space,
            config.search.predictor_type,
            config.seed,
        )
    elif config_type == "statistics":
        config.save = "{}/{}/{}/{}/{}".format(
            config.out_dir,
            config.search_space,
            config.dataset,
            "statistics",
            config.seed,
        )
    elif config_type == "zc":
        if not hasattr(config, 'search'):
            config.search = copy.deepcopy(config)
        if not hasattr(config, 'evaluation'):
            config.evaluation = CfgNode()

        if args.datapath is not None:
            config.train_data_file = os.path.join(args.datapath, 'train.json')
            config.test_data_file = os.path.join(args.datapath, 'test.json')
        else:
            config.train_data_file = None
            config.test_data_file = None

        config.save = "{}/{}/{}/{}/{}/{}".format(
            config.out_dir,
            config.config_type,
            config.search_space,
            config.dataset,
            config.predictor,
            config.seed,
        )

    else:
        print("invalid config type in utils/utils.py")

    config.data = "{}/data".format(get_project_root())

    create_exp_dir(config.save)
    create_exp_dir(config.save + "/search")  # required for the checkpoints
    create_exp_dir(config.save + "/eval")

    return config


class TensorDatasetWithTrans(Dataset):
    """
    TensorDataset with support of transforms.
    """

    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


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
        path = os.path.join(
            config.save, "search" if search else "eval", "last_checkpoint"
        )
        with open(path, "r") as f:
            checkpoint_name = f.readline()
        return os.path.join(
            config.save, "search" if search else "eval", checkpoint_name
        )
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
    return (
        np.sum(
            np.prod(v.size())
            for name, v in model.named_parameters()
            if "auxiliary" not in name
        )
        / 1e6
    )


def log_args(args):
    """
    Log the args in a nice way.
    """
    for arg, val in args.items():
        logger.info(arg + "." * (50 - len(arg) - len(str(val))) + str(val))


def create_exp_dir(path):
    """
    Create the experiment directories.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    logger.info("Experiment dir : {}".format(path))


def cross_validation(
        xtrain, ytrain, predictor, split_indices, score_metric="kendalltau"
):
    validation_score = []

    for train_indices, validation_indices in split_indices:
        xtrain_i = [xtrain[j] for j in train_indices]
        ytrain_i = [ytrain[j] for j in train_indices]
        xval_i = [xtrain[j] for j in train_indices]
        yval_i = [ytrain[j] for j in train_indices]

        predictor.fit(xtrain_i, ytrain_i)
        ypred_i = predictor.query(xval_i)

        # If the predictor is an ensemble, take the mean
        if len(ypred_i.shape) > 1:
            ypred_i = np.mean(ypred_i, axis=0)

        # use Pearson correlation to be the metric -> maximise Pearson correlation
        if score_metric == "pearson":
            score_i = np.abs(np.corrcoef(yval_i, ypred_i)[1, 0])
        elif score_metric == "mae":
            score_i = np.mean(abs(ypred_i - yval_i))
        elif score_metric == "rmse":
            score_i = metrics.mean_squared_error(
                yval_i, ypred_i, squared=False)
        elif score_metric == "spearman":
            score_i = stats.spearmanr(yval_i, ypred_i)[0]
        elif score_metric == "kendalltau":
            score_i = stats.kendalltau(yval_i, ypred_i)[0]
        elif score_metric == "kt_2dec":
            score_i = stats.kendalltau(
                yval_i, np.round(ypred_i, decimals=2))[0]
        elif score_metric == "kt_1dec":
            score_i = stats.kendalltau(
                yval_i, np.round(ypred_i, decimals=1))[0]

        validation_score.append(score_i)

    return np.mean(validation_score)


def generate_kfold(n, k):
    """
    Input:
        n: number of training examples
        k: number of folds
    Returns:
        kfold_indices: a list of len k. Each entry takes the form
        (training indices, validation indices)
    """
    assert k >= 2
    kfold_indices = []

    indices = np.array(range(n))
    fold_size = n // k

    fold_indices = [
        indices[i * fold_size: (i + 1) * fold_size] for i in range(k - 1)]
    fold_indices.append(indices[(k - 1) * fold_size:])

    for i in range(k):
        training_indices = [fold_indices[j] for j in range(k) if j != i]
        validation_indices = fold_indices[i]
        kfold_indices.append(
            (np.concatenate(training_indices), validation_indices))

    return kfold_indices


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class NamedAverageMeter:
    """Computes and stores the average and current value, ported from naszilla repo"""

    def __init__(self, name, fmt=":f"):
        """
        Initialization of AverageMeter
        Parameters
        ----------
        name : str
            Name to display.
        fmt : str
            Format string to print the values.
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = "{name}: {avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)


class AverageMeterGroup:
    """Average meter group for multiple average meters, ported from Naszilla repo."""

    def __init__(self):
        self.meters = OrderedDict()

    def update(self, data, n=1):
        for k, v in data.items():
            if k not in self.meters:
                self.meters[k] = NamedAverageMeter(k, ":4f")
            self.meters[k].update(v, n=n)

    def __getattr__(self, item):
        return self.meters[item]

    def __getitem__(self, item):
        return self.meters[item]

    def __str__(self):
        return "  ".join(str(v) for v in self.meters.values())

    def summary(self):
        return "  ".join(v.summary() for v in self.meters.values())


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
            self.logger.info(
                "No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(path))
        if not os.path.isfile(path):
            path = PathManager.get_local_path(path)
            assert os.path.isfile(
                path), "Checkpoint {} not found!".format(path)

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
