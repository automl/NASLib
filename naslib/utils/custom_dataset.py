import numpy as np
import torch
import abc

from typing import List
from torchvision.transforms import Compose
from torch.utils.data import Dataset


class CustomDataset():
    def __init__(self, config, mode="train"):
        self.data = config.data
        self.dataset = config.dataset
        self.seed = config.search.seed
        self.batch_size = config.batch_size if hasattr(config, "batch_size") else config.search.batch_size
        self.train_portion = config.train_portion if hasattr(config, "train_portion") else config.search.train_portion
        self.config = config.search if mode == "train" else config.evaluation

    @abc.abstractmethod
    def get_transforms(self, config) -> List[Compose]:
        """
        Get the transform that your dataset uses.

        config: CfgNode 
        return -> train_transform, valid_transform
        """
        raise NotImplementedError('')

    @abc.abstractmethod
    def get_data(self, data, train_transform, valid_transform) -> List[Dataset]:
        """
        Get the data required to create the loaders 

        data: root directory of the dataset. 
            See https://pytorch.org/vision/stable/datasets.html for how to store 
            torch Datasets
        train_transform: torchvision.transform.Compose object for train loader
        valid_transform: torchvision.transform.Compose object for valid loader
        
        return -> train_data, test_data
        """
        raise NotImplementedError('')

    def get_loaders(self):
        train_transform, valid_transform = self.get_transforms(self.config)
        train_data, test_data = self.get_data(self.data, train_transform, valid_transform)

        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(self.train_portion * num_train))

        train_queue = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True,
            num_workers=0,
            worker_init_fn=np.random.seed(self.seed+1),
        )

        valid_queue = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
            pin_memory=True,
            num_workers=0,
            worker_init_fn=np.random.seed(self.seed),
        )

        test_queue = torch.utils.data.DataLoader(
            test_data,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
            worker_init_fn=np.random.seed(self.seed),
        ) 

        return train_queue, valid_queue, test_queue, train_transform, valid_transform
