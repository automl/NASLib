# taskonomy_dataset.py defines the TaskonomyDataset class, 
# which is used to load the data for the taskonomy dataset.

import os.path as osp
import torch
from skimage import io
from torch.utils.data import Dataset

from . import load_ops

DOMAIN_DATA_SOURCE = {
    'rgb': ('rgb', 'png'),
    'autoencoder': ('rgb', 'png'),
    'class_object': ('class_object', 'npy'),
    'class_scene': ('class_scene', 'npy'),
    'normal': ('normal', 'png'),
    'room_layout': ('room_layout', 'npy'),
    'segmentsemantic': ('segmentsemantic', 'png'),
    'jigsaw': ('rgb', 'png'),
}

class TaskonomyDataset(Dataset):
    def __init__(self, json_path, dataset_dir, domain, target_load_fn, target_load_kwargs=None, transform=None):
        """
        Loading Taskonomy Datasets.

        Args:
            json_path (string): /path/to/json_file for train/val/test_filenames (specify which buildings to include)
            dataset_dir (string): Directory with all the images.
            domain (string): Domain of the dataset.
            target_load_fn (function): Function to load the target data.
            target_load_kwargs (dict): Keyword arguments for the target load function.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.json_path = json_path
        self.dataset_dir = dataset_dir
        self.domain = domain
        self.all_templates = get_all_templates(dataset_dir, json_path)
        self.target_load_fn = target_load_fn
        self.target_load_kwargs = target_load_kwargs
        self.transform = transform
    
    def __len__(self):
        return len(self.all_templates)

    def __getitem__(self, idx):
        try:
            if torch.is_tensor(idx):
                idx = idx.tolist()
            template = osp.join(self.dataset_dir, self.all_templates[idx])
            image = io.imread('.'.join([template.format(domain='rgb'), 'png']))
            label = self.get_label(template)
            sample = {'image': image, 'label': label}
            if self.transform:
                sample = self.transform(sample)
        except:
            template = osp.join(self.dataset_dir, self.all_templates[idx])
            raise Exception('Error loading image: {}'.format('.'.join([template.format(domain='rgb'), 'png'])))
        sample = [sample['image'], sample['label']]
        return sample

    def get_label(self, template):
        domain, file_type = DOMAIN_DATA_SOURCE[self.domain]
        label_path = '.'.join([template.format(domain=domain), file_type])
        label = self.target_load_fn(label_path, **self.target_load_kwargs)
        return label


def get_all_templates(dataset_dir, filenames_path):
    """
    Get all the templates in the dataset.

    Args:
        dataset_dir (string): Directory with all the images.
        filenames_path (string): /path/to/json_file for train/val/test_filenames (specify which buildings to include)
    """
    building_lists = load_ops.read_json(filenames_path)
    all_template_paths = []
    for building in building_lists:
        all_template_paths += load_ops.read_json(osp.join(dataset_dir, f"{building}.json"))
    for i, path in enumerate(all_template_paths):
        f_split = path.split('.')
        if f_split[-1] in ['npy', 'png']:
            all_template_paths[i] = '.'.join(f_split[:-1])
    return all_template_paths


def get_datasets(cfg):
    """Getting the train/val/test dataset"""
    train_data = TaskonomyDataset(osp.join(cfg['data_split_dir'], cfg['train_filenames']),
                                cfg['dataset_dir'], cfg['task_name'], cfg['target_load_fn'], 
                                target_load_kwargs=cfg['target_load_kwargs'], 
                                transform=cfg['train_transform_fn'])
    val_data = TaskonomyDataset(osp.join(cfg['data_split_dir'], cfg['val_filenames']),
                                cfg['dataset_dir'], cfg['task_name'], cfg['target_load_fn'], 
                                target_load_kwargs=cfg['target_load_kwargs'], 
                                transform=cfg['val_transform_fn'])
    test_data = TaskonomyDataset(osp.join(cfg['data_split_dir'], cfg['test_filenames']),
                                cfg['dataset_dir'], cfg['task_name'], cfg['target_load_fn'], 
                                target_load_kwargs=cfg['target_load_kwargs'], 
                                transform=cfg['test_transform_fn'])
    return train_data, val_data, test_data
