import os
import numpy as np
import torch
from pathlib import Path

import torchvision.datasets as dset
import torchvision.transforms as transforms

from .taskonomy_dataset import get_datasets
from . import load_ops


def get_project_root() -> Path:
    """
    Returns the root path of the project.
    """
    return Path(__file__).parent.parent


def get_train_val_loaders(config, mode="train"):
    """
    Constructs the dataloaders and transforms for training, validation and test data.
    """
    data = config.data
    dataset = config.dataset
    seed = config.search.seed
    batch_size = config.batch_size if hasattr(
        config, "batch_size") else config.search.batch_size
    train_portion = config.train_portion if hasattr(
        config, "train_portion") else config.search.train_portion
    config = config.search if mode == "train" else config.evaluation
    if dataset == "cifar10":
        train_transform, valid_transform = _data_transforms_cifar10(config)
        train_data = dset.CIFAR10(
            root=data, train=True, download=True, transform=train_transform
        )
        test_data = dset.CIFAR10(
            root=data, train=False, download=True, transform=valid_transform
        )
    elif dataset == "cifar100":
        train_transform, valid_transform = _data_transforms_cifar100(config)
        train_data = dset.CIFAR100(
            root=data, train=True, download=True, transform=train_transform
        )
        test_data = dset.CIFAR100(
            root=data, train=False, download=True, transform=valid_transform
        )
    elif dataset == "svhn":
        train_transform, valid_transform = _data_transforms_svhn(config)
        train_data = dset.SVHN(
            root=data, split="train", download=True, transform=train_transform
        )
        test_data = dset.SVHN(
            root=data, split="test", download=True, transform=valid_transform
        )
    elif dataset == "ImageNet16-120":
        from naslib.utils.DownsampledImageNet import ImageNet16

        train_transform, valid_transform = _data_transforms_ImageNet_16_120(
            config)
        data_folder = os.path.join(data, dataset)
        train_data = ImageNet16(
            root=data_folder,
            train=True,
            transform=train_transform,
            use_num_of_class_only=120,
        )
        test_data = ImageNet16(
            root=data_folder,
            train=False,
            transform=valid_transform,
            use_num_of_class_only=120,
        )
    elif dataset == 'ninapro':
        from naslib.utils.ninapro_dataset import NinaPro, ninapro_transform

        train_transform, valid_transform = ninapro_transform(config)
        data_folder = os.path.join(data, dataset)
        train_data = NinaPro(data_folder, split="train", transform=train_transform)
        test_data = NinaPro(data_folder, split="test", transform=valid_transform)
    elif dataset == "darcyflow":
        from naslib.utils.darcyflow_dataset import load_darcyflow_data, darcyflow_transform

        train_transform, valid_transform = darcyflow_transform(config)

        data_folder = os.path.join(data, dataset)
        train_data, test_data = load_darcyflow_data(data_folder)
    elif dataset == 'jigsaw':
        cfg = get_jigsaw_configs()

        try:
            train_data, val_data, test_data = get_datasets(cfg)
        except:
            raise FileNotFoundError(
                "The jigsaw dataset has not been downloaded, run scripts/bash_scripts/download_data.sh")

        train_transform = cfg['train_transform_fn']
        valid_transform = cfg['val_transform_fn']

    elif dataset == 'class_object':
        cfg = get_class_object_configs()

        try:
            train_data, val_data, test_data = get_datasets(cfg)
        except:
            raise FileNotFoundError(
                "The class_object dataset has not been downloaded, run scripts/bash_scripts/download_data.sh")

        train_transform = cfg['train_transform_fn']
        valid_transform = cfg['val_transform_fn']

    elif dataset == 'class_scene':
        cfg = get_class_scene_configs()

        try:
            train_data, val_data, test_data = get_datasets(cfg)
        except:
            raise FileNotFoundError(
                "The class_scene dataset has not been downloaded, run scripts/bash_scripts/download_data.sh")

        train_transform = cfg['train_transform_fn']
        valid_transform = cfg['val_transform_fn']

    elif dataset == 'autoencoder':
        cfg = get_autoencoder_configs()

        try:
            train_data, val_data, test_data = get_datasets(cfg)
        except:
            raise FileNotFoundError(
                "The autoencoder dataset has not been downloaded, run scripts/bash_scripts/download_data.sh")

        train_transform = cfg['train_transform_fn']
        valid_transform = cfg['val_transform_fn']

    elif dataset == 'segmentsemantic':
        cfg = get_segmentsemantic_configs()

        train_data, val_data, test_data = get_datasets(cfg)

        train_transform = cfg['train_transform_fn']
        valid_transform = cfg['val_transform_fn']

    elif dataset == 'normal':
        cfg = get_normal_configs()

        train_data, val_data, test_data = get_datasets(cfg)

        train_transform = cfg['train_transform_fn']
        valid_transform = cfg['val_transform_fn']

    elif dataset == 'room_layout':
        cfg = get_room_layout_configs()

        train_data, val_data, test_data = get_datasets(cfg)

        train_transform = cfg['train_transform_fn']
        valid_transform = cfg['val_transform_fn']

    else:
        # 3 things for datasets
        # train_data, val_data, test_data
        # Train Tansform, validation transform functions
        raise ValueError("Unknown dataset: {}".format(dataset))

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True,
        num_workers=0,
        worker_init_fn=np.random.seed(seed + 1),
    )

    valid_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(
            indices[split:num_train]),
        pin_memory=True,
        num_workers=0,
        worker_init_fn=np.random.seed(seed),
    )

    test_queue = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        worker_init_fn=np.random.seed(seed),
    )

    return train_queue, valid_queue, test_queue, train_transform, valid_transform


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    if hasattr(args, 'cutout') and args.cutout:
        train_transform.transforms.append(
            Cutout(args.cutout_length, args.cutout_prob))

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    return train_transform, valid_transform


def _data_transforms_svhn(args):
    SVHN_MEAN = [0.4377, 0.4438, 0.4728]
    SVHN_STD = [0.1980, 0.2010, 0.1970]

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(SVHN_MEAN, SVHN_STD),
        ]
    )
    if args.cutout:
        train_transform.transforms.append(
            Cutout(args.cutout_length, args.cutout_prob))

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(SVHN_MEAN, SVHN_STD),
        ]
    )
    return train_transform, valid_transform


def _data_transforms_cifar100(args):
    CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
    CIFAR_STD = [0.2673, 0.2564, 0.2762]

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    if args.cutout:
        train_transform.transforms.append(
            Cutout(args.cutout_length, args.cutout_prob))

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    return train_transform, valid_transform


def _data_transforms_ImageNet_16_120(args):
    IMAGENET16_MEAN = [x / 255 for x in [122.68, 116.66, 104.01]]
    IMAGENET16_STD = [x / 255 for x in [63.22, 61.26, 65.09]]

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(16, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET16_MEAN, IMAGENET16_STD),
        ]
    )
    if args.cutout:
        train_transform.transforms.append(
            Cutout(args.cutout_length, args.cutout_prob))

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET16_MEAN, IMAGENET16_STD),
        ]
    )
    return train_transform, valid_transform


def get_jigsaw_configs():
    cfg = {}

    cfg['task_name'] = 'jigsaw'

    cfg['input_dim'] = (255, 255)
    cfg['target_num_channels'] = 9

    cfg['dataset_dir'] = os.path.join(
        get_project_root(), "data", "taskonomydata_mini")
    cfg['data_split_dir'] = cfg['dataset_dir']

    cfg['train_filenames'] = 'train_split.json'
    cfg['val_filenames'] = 'val_split.json'
    cfg['test_filenames'] = 'test_split.json'

    cfg['target_dim'] = 1000
    cfg['target_load_fn'] = load_ops.random_jigsaw_permutation
    cfg['target_load_kwargs'] = {'classes': cfg['target_dim']}

    cfg['train_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.RandomHorizontalFlip(0.5),
        load_ops.ColorJitter(brightness=0.2, contrast=0.2,
                             saturation=0.2, hue=0.1),
        load_ops.RandomGrayscale(0.3),
        load_ops.MakeJigsawPuzzle(classes=cfg['target_dim'], mode='max', tile_dim=(64, 64), centercrop=0.9, norm=False,
                                  totensor=True),
    ])

    cfg['val_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.RandomHorizontalFlip(0.5),
        load_ops.ColorJitter(brightness=0.2, contrast=0.2,
                             saturation=0.2, hue=0.1),
        load_ops.RandomGrayscale(0.3),
        load_ops.MakeJigsawPuzzle(classes=cfg['target_dim'], mode='max', tile_dim=(64, 64), centercrop=0.9, norm=False,
                                  totensor=True),
    ])

    cfg['test_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.RandomHorizontalFlip(0.5),
        load_ops.ColorJitter(brightness=0.2, contrast=0.2,
                             saturation=0.2, hue=0.1),
        load_ops.RandomGrayscale(0.3),
        load_ops.MakeJigsawPuzzle(classes=cfg['target_dim'], mode='max', tile_dim=(64, 64), centercrop=0.9, norm=False,
                                  totensor=True),
    ])
    return cfg


def get_class_object_configs():
    cfg = {}

    cfg['task_name'] = 'class_object'

    cfg['input_dim'] = (256, 256)
    cfg['input_num_channels'] = 3

    cfg['dataset_dir'] = os.path.join(
        get_project_root(), "data", "taskonomydata_mini")
    cfg['data_split_dir'] = cfg['dataset_dir']

    cfg['train_filenames'] = 'train_split.json'
    cfg['val_filenames'] = 'val_split.json'
    cfg['test_filenames'] = 'test_split.json'

    cfg['target_dim'] = 75

    cfg['target_load_fn'] = load_ops.load_class_object_logits

    cfg['target_load_kwargs'] = {'selected': True if cfg['target_dim'] < 1000 else False,
                                 'final5k': True if cfg['data_split_dir'].split('/')[-1] == 'final5k' else False}

    cfg['demo_kwargs'] = {'selected': True if cfg['target_dim'] < 1000 else False,
                          'final5k': True if cfg['data_split_dir'].split('/')[-1] == 'final5k' else False}

    cfg['normal_params'] = {'mean': [0.5224, 0.5222,
                                     0.5221], 'std': [0.2234, 0.2235, 0.2236]}

    cfg['train_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.RandomHorizontalFlip(0.5),
        load_ops.ColorJitter(brightness=0.2, contrast=0.2,
                             saturation=0.2, hue=0.1),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])

    cfg['val_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])

    cfg['test_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])
    return cfg


def get_class_scene_configs():
    cfg = {}

    cfg['task_name'] = 'class_scene'

    cfg['input_dim'] = (256, 256)
    cfg['input_num_channels'] = 3

    cfg['dataset_dir'] = os.path.join(
        get_project_root(), "data", "taskonomydata_mini")
    cfg['data_split_dir'] = cfg['dataset_dir']

    cfg['train_filenames'] = 'train_split.json'
    cfg['val_filenames'] = 'val_split.json'
    cfg['test_filenames'] = 'test_split.json'

    cfg['target_dim'] = 47

    cfg['target_load_fn'] = load_ops.load_class_scene_logits

    cfg['target_load_kwargs'] = {'selected': True if cfg['target_dim'] < 365 else False,
                                 'final5k': True if cfg['data_split_dir'].split('/')[-1] == 'final5k' else False}

    cfg['demo_kwargs'] = {'selected': True if cfg['target_dim'] < 365 else False,
                          'final5k': True if cfg['data_split_dir'].split('/')[-1] == 'final5k' else False}

    cfg['normal_params'] = {'mean': [0.5224, 0.5222,
                                     0.5221], 'std': [0.2234, 0.2235, 0.2236]}

    cfg['train_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.RandomHorizontalFlip(0.5),
        load_ops.ColorJitter(brightness=0.2, contrast=0.2,
                             saturation=0.2, hue=0.1),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])

    cfg['val_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])

    cfg['test_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])
    return cfg


def get_autoencoder_configs():
    cfg = {}

    cfg['task_name'] = 'autoencoder'

    cfg['input_dim'] = (256, 256)
    cfg['input_num_channels'] = 3

    cfg['target_dim'] = (256, 256)
    cfg['target_channel'] = 3

    cfg['dataset_dir'] = os.path.join(
        get_project_root(), "data", "taskonomydata_mini")
    cfg['data_split_dir'] = cfg['dataset_dir']

    cfg['train_filenames'] = 'train_split.json'
    cfg['val_filenames'] = 'val_split.json'
    cfg['test_filenames'] = 'test_split.json'

    cfg['target_load_fn'] = load_ops.load_raw_img_label
    cfg['target_load_kwargs'] = {}

    cfg['normal_params'] = {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}

    cfg['train_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.RandomHorizontalFlip(0.5),
        load_ops.ColorJitter(brightness=0.2, contrast=0.2,
                             saturation=0.2, hue=0.1),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])

    cfg['val_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])

    cfg['test_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])
    return cfg


def get_segmentsemantic_configs():
    cfg = {}

    cfg['task_name'] = 'segmentsemantic'

    cfg['input_dim'] = (256, 256)
    cfg['input_num_channels'] = 3

    cfg['target_dim'] = (256, 256)
    cfg['target_num_channel'] = 17

    cfg['dataset_dir'] = os.path.join(
        get_project_root(), "data", "taskonomydata_mini")
    cfg['data_split_dir'] = cfg['dataset_dir']

    cfg['train_filenames'] = 'train_split.json'
    cfg['val_filenames'] = 'val_split.json'
    cfg['test_filenames'] = 'test_split.json'

    cfg['target_load_fn'] = load_ops.semantic_segment_label
    cfg['target_load_kwargs'] = {}

    cfg['normal_params'] = {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}

    cfg['train_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.RandomHorizontalFlip(0.5),
        load_ops.ColorJitter(brightness=0.2, contrast=0.2,
                             saturation=0.2, hue=0.1),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])

    cfg['val_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])

    cfg['test_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])
    return cfg


def get_normal_configs():
    cfg = {}

    cfg['task_name'] = 'normal'

    cfg['input_dim'] = (256, 256)
    cfg['input_num_channels'] = 3

    cfg['target_dim'] = (256, 256)
    cfg['target_channel'] = 3

    cfg['dataset_dir'] = os.path.join(
        get_project_root(), "data", "taskonomydata_mini")
    cfg['data_split_dir'] = cfg['dataset_dir']

    cfg['train_filenames'] = 'train_split.json'
    cfg['val_filenames'] = 'val_split.json'
    cfg['test_filenames'] = 'test_split.json'

    cfg['target_load_fn'] = load_ops.load_raw_img_label
    cfg['target_load_kwargs'] = {}

    cfg['normal_params'] = {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}

    cfg['train_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        # load_ops.RandomHorizontalFlip(0.5),
        # load_ops.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])

    cfg['val_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])

    cfg['test_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])
    return cfg


def get_room_layout_configs():
    cfg = {}

    cfg['task_name'] = 'room_layout'

    cfg['input_dim'] = (256, 256)
    cfg['input_num_channels'] = 3

    cfg['target_dim'] = 9

    cfg['dataset_dir'] = os.path.join(
        get_project_root(), "data", "taskonomydata_mini")
    cfg['data_split_dir'] = os.path.join(
        get_project_root(), "data", "final5K_splits")

    cfg['train_filenames'] = 'train_filenames_final5k.json'
    cfg['val_filenames'] = 'val_filenames_final5k.json'
    cfg['test_filenames'] = 'test_filenames_final5k.json'

    cfg['target_load_fn'] = load_ops.point_info2room_layout
    # cfg['target_load_fn'] = load_ops.room_layout
    cfg['target_load_kwargs'] = {}

    cfg['normal_params'] = {'mean': [0.5224, 0.5222,
                                     0.5221], 'std': [0.2234, 0.2235, 0.2236]}

    cfg['train_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        # load_ops.RandomHorizontalFlip(0.5),
        load_ops.ColorJitter(brightness=0.2, contrast=0.2,
                             saturation=0.2, hue=0.1),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])

    cfg['val_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])

    cfg['test_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])
    return cfg


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

            mask[y1:y2, x1:x2] = 0.0
            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img *= mask
        return img
