import os
from typing import Optional, Tuple, Union, List

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets
from torchvision.datasets import VisionDataset
import torch
from . import preproc


class DataModule(LightningDataModule):
    def __init__(self, dataset, data_dir='datasets', split_train: bool = False, return_train_val: bool = True,
                 cutout_length=0, batch_size=64, workers: Optional[int] = None,
                 train_transforms=None, val_transforms=None, test_transforms=None, dims=None):
        self.dataset_name: str = dataset.lower()
        self.data_dir: str = data_dir
        self.split_train: bool = split_train
        self.return_train_val: bool = return_train_val
        self.train_indices: List[int] = []
        self.valid_indices: List[int] = []

        self.cutout_length: int = cutout_length
        self.batch_size: int = batch_size
        self.workers: int = max(os.cpu_count() - 1, 1) if workers is None else workers

        if self.dataset_name == 'cifar10':          dataset_class, n_classes = datasets.CIFAR10, 10
        elif self.dataset_name == 'cifar100':       dataset_class, n_classes = datasets.CIFAR100, 100
        elif self.dataset_name == 'imagenet':       dataset_class, n_classes = datasets.ImageNet, 1000
        elif self.dataset_name == 'mnist':          dataset_class, n_classes = datasets.MNIST, 10
        elif self.dataset_name == 'fashionmnist':   dataset_class, n_classes = datasets.FashionMNIST, 10
        else:
            raise ValueError(self.dataset_name)

        self.dataset_class: type(VisionDataset) = dataset_class
        self.n_classes: int = n_classes
        if train_transforms is None or val_transforms is None:
            train_transforms, val_transforms = preproc.data_transforms(self.dataset_name, self.cutout_length)
            test_transforms = val_transforms
        self.train_data: Optional[VisionDataset] = None
        self.valid_data: Optional[VisionDataset] = None
        self.shape: Tuple = ()
        self.input_channels: int = 0
        self.input_size: int = 0

        super().__init__(train_transforms, val_transforms, test_transforms, dims)

    def prepare_data(self, *args, **kwargs):
        self.dataset_class(root=self.data_dir, train=True, download=True)
        self.dataset_class(root=self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        self.train_data = self.dataset_class(root=self.data_dir, train=True, download=True, transform=self.train_transforms)
        self.valid_data = self.dataset_class(root=self.data_dir, train=False, download=True, transform=self.val_transforms)

        if self.split_train:
            n_train = len(self.train_data)
            split = n_train // 2
            indices = list(range(n_train))
            self.train_indices = indices[:split]
            self.valid_indices = indices[split:]
        else:
            self.train_indices = list(range(len(self.train_data)))
            self.valid_indices = list(range(len(self.valid_data)))

        # assuming shape is NHW or NHWC
        self.shape = self.train_data.data.shape
        self.input_channels = 3 if len(self.shape) == 4 else 1
        assert self.shape[1] == self.shape[2], "Not expected shape = {}".format(self.shape)
        self.input_size = self.shape[1]
        self.dims = self.train_data[0][0].shape

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        if self.split_train:
            train_subset, val_subset = torch.utils.data.random_split(self.train_data, [int(0.5*len(self.train_data)), int(0.5*len(self.train_data))])
            train_loader = DataLoader(train_subset, batch_size=self.batch_size,
                                      num_workers=self.workers, pin_memory=True)
            valid_loader = DataLoader(val_subset, batch_size=self.batch_size,
                                      num_workers=self.workers, pin_memory=True)
            if self.return_train_val:
                return [train_loader, valid_loader]
            return train_loader

        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.workers, pin_memory=True)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        if self.split_train:
            return DataLoader(self.train_data, batch_size=self.batch_size,
                              sampler=SubsetRandomSampler(self.valid_indices),
                              num_workers=self.workers, pin_memory=True)

        return DataLoader(self.valid_data, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.workers, pin_memory=True)
