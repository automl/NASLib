import torchvision.datasets as dset

from naslib.utils.custom_dataset import CustomDataset
from naslib.utils.dataset import _data_transforms_cifar10
from naslib.utils import get_config_from_args


if __name__ == '__main__':
    class MyCustomDataset(CustomDataset):
        def __init__(self, config, mode="train"):
            super().__init__(config, mode)


        def get_transforms(self, config):
            train_transform, valid_transform = _data_transforms_cifar10(config)
            return train_transform, valid_transform
            

        def get_data(self, data, train_transform, valid_transform):
            train_data = dset.CIFAR10(
                root=data, train=True, download=True, transform=train_transform
            )
            test_data = dset.CIFAR10(
                root=data, train=False, download=True, transform=valid_transform
            )

            return train_data, test_data
            
    config = get_config_from_args()
    dataset = MyCustomDataset(config)
    train_queue, valid_queue, test_queue, train_transform, valid_transform = dataset.get_loaders()
    