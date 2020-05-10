import os
import sys
import logging
import numpy as np
import torch.utils
import torchvision.datasets as dset

from copy import copy

from naslib.utils import utils


class Parser(object):
    def __init__(self, config_file):
        self.args = utils.config_parser(config_file)
        utils.print_args(self.args)

        self.args._save = copy(self.args.save)
        self.args.save = '{}/{}'.format(self.args.save,
                                        self.args.dataset)

        utils.create_exp_dir(self.args.save)

        if self.args.dataset != 'cifar100':
            self.args.n_classes = 10
        else:
            self.args.n_classes = 100


    @property
    def config(self):
        return self.args


    def get_train_val_loaders(self):
        if self.args.dataset == 'cifar10':
            train_transform, valid_transform = utils._data_transforms_cifar10(self.args)
            train_data = dset.CIFAR10(root=self.args.data, train=True, download=True, transform=train_transform)
            test_data = dset.CIFAR10(root=self.args.data, train=False, download=True, transform=valid_transform)
        elif self.args.dataset == 'cifar100':
            train_transform, valid_transform = utils._data_transforms_cifar100(self.args)
            train_data = dset.CIFAR100(root=self.args.data, train=True, download=True, transform=train_transform)
            test_data = dset.CIFAR100(root=self.args.data, train=False, download=True, transform=valid_transform)
        elif self.args.dataset == 'svhn':
            train_transform, valid_transform = utils._data_transforms_svhn(self.args)
            train_data = dset.SVHN(root=self.args.data, split='train', download=True, transform=train_transform)
            test_data = dset.SVHN(root=self.args.data, split='test', download=True, transform=valid_transform)

        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(self.args.train_portion * num_train))

        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=self.args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True, num_workers=0, worker_init_fn=np.random.seed(self.config.seed))

        valid_queue = torch.utils.data.DataLoader(
            train_data, batch_size=self.args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
            pin_memory=True, num_workers=0, worker_init_fn=np.random.seed(self.config.seed))

        test_queue = torch.utils.data.DataLoader(
            test_data, batch_size=self.args.batch_size, shuffle=False,
            pin_memory=True, num_workers=0)


        return train_queue, valid_queue, test_queue, train_transform, valid_transform
