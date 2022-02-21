import os
import sys
import json
import random
import numpy as np
from PIL import Image
import collections
import torch
import torchvision.transforms as T
from skimage import io
from torchvision.transforms import functional as F

if sys.version_info < (3, 3):
    sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable
from pathlib import Path

lib_dir = (Path(__file__).parent / '..').resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
# from data.synset import synset as raw_synset


# Helper fns
def read_json(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

# Image transform fns
_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}

TASK_NAMES = [
    "autoencoder",
    "class_object",
    "class_scene",
    "normal",
    "jigsaw",
    "room_layout",
    "segmentsemantic"
]

class Compose(T.Compose):
    def __init__(self, task_name, transforms):
        self.transforms = transforms
        self.task_name = task_name
        assert task_name in TASK_NAMES, "task_name must be one of {}".format(TASK_NAMES)

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample, self.task_name)
        return sample

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, length):
        self.length = int(length)

    def cutout(self, img):
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
    
    def __repr__(self) -> str:
        return '{name}(length={length})'.format(name=self.__class__.__name__, **self.__dict__)

    def __call__(self, sample, task_name):
        image, label = sample['image'], sample['label']
        if task_name in ['class_object', 'class_scene', 'room_layout']:
            return {
                'image': self.cutout(image),
                'label': label
            }
        else:
            raise ValueError(f"task name {task_name} not supported!")

class Resize(T.Resize):
    def __init__(self, input_size, target_size=None, interpolation=Image.BILINEAR):
        assert isinstance(input_size, int) or (isinstance(input_size, Iterable) and len(input_size) == 2)
        if target_size:
            assert isinstance(target_size, int) or (isinstance(target_size, Iterable) and len(target_size) == 2)
        self.input_size = input_size
        self.target_size = target_size if target_size else input_size
        self.interpolation = interpolation

    def __call__(self, sample, task_name):
        image, label = sample['image'], sample['label']
        if task_name in ['autoencoder', 'normal']:
            return {
                'image': F.resize(image, self.input_size, self.interpolation),
                'label': F.resize(label, self.target_size, self.interpolation),
            }
        elif task_name == 'segmentsemantic':
            return {
                'image': F.resize(image, self.input_size, self.interpolation),
                'label': F.resize(label, self.target_size, Image.NEAREST),
            }
        elif task_name in ['class_object', 'class_scene', 'room_layout', 'jigsaw']:
            return {
                'image': F.resize(image, self.input_size, self.interpolation),
                'label': label,
            }
        else:
            raise ValueError(f"task name {task_name} not supported!")

    def __repr__(self) -> str:
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + f"(input_size={self.input_size}, target_size={self.target_size}, interpolation={interpolate_str}"
    

class ToPILImage(T.ToPILImage):
    
    def __init__(self, mode=None):
        self.mode = mode
        
    def __call__(self, sample, task_name):
        
        image, label = sample['image'], sample['label']
        if task_name in ['autoencoder', 'normal']:
            return {'image': F.to_pil_image(image, self.mode), 'label': F.to_pil_image(label, self.mode)}
        elif task_name == 'segmentsemantic':
            return {'image': F.to_pil_image(image, self.mode), 'label': F.to_pil_image(label, self.mode)}
        elif task_name in ['class_scene', 'class_object', 'jigsaw', 'room_layout']:
            return {'image': F.to_pil_image(image, self.mode), 'label': label}
        else:
            raise ValueError(f'task name {task_name} not available!')

            
class ToTensor(T.ToTensor):
    
    def __init__(self, new_scale=None):
        self.new_scale = new_scale
        
    def __call__(self, sample, task_name):
        
        image, label = sample['image'], sample['label']
        if task_name in ['autoencoder', 'normal']:
            image = F.to_tensor(image).float()
            label = F.to_tensor(label).float()
            if self.new_scale:
                min_val, max_val = self.new_scale
                label *= (max_val - min_val)
                label += min_val
        elif task_name == 'segmentsemantic':
            image = F.to_tensor(image).float()
            label = torch.tensor(np.array(label), dtype=torch.unit8)
        elif task_name in ['class_scene', 'class_object', 'room_layout']:
            image = F.to_tensor(image).float()
            label = torch.FloatTensor(label)
        else:
            raise ValueError(f'task name {task_name} not available!')
            
        if self.new_scale:
            min_val, max_val = self.new_scale
            image *= (max_val - min_val)
            image += min_valkm
        return {'image': image, 'label': label}
    
    
class Normalize(T.Normalize):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace
        
    def __call__(self, sample, task_name):
        
        tensor, label = sample['image'], sample['label']
        if task_name in ['autoencoder']:
            return {'image': F.normalize(tensor, self.mean, self.std, self.inplace), 
                    'label': F.normalize(tensor, self.mean, self.std, self.inplace)}
        elif task_name in ['normal', 'segmentsemantic']:
            return {'image': F.normalize(tensor, self.mean, self.std, self.inplace), 
                    'label': label}
        elif task_name in ['class_scene', 'class_object', 'room_layout']:
            return {'image': F.normalize(tensor, self.mean, self.std, self.inplace), 
                    'label': label}
        else:
            raise ValueError(f'task name {task_name} not available!')

                  

class RandomHorizontalFlip(T.RandomHorizontalFlip):
    
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, sample, task_name):
        random_num = random.random()
        image, label = sample['image'], sample['label']
        if random_num < self.p:
            if task_name in ['autoencoder', 'segmentsemantic']:
                return {'image': F.hflip(image), 'label': F.hflip(label)}
            elif task_name in ['class_scene', 'class_object', 'jigsaw']:
                return {'image': F.hflip(image), 'label': label}
            elif task_name in ['normal', 'room_layout']:
                raise ValueError(f'task name {task_name} not available!')
            else:
                raise ValueError(f'task name {task_name} not available!')
        else:
            return sample

class RandomGrayscale(T.RandomGrayscale):
    
    def __init__(self, p=0.1):
        self.p = p
        
    def __call__(self, sample, task_name):
        
        image, label = sample['image'], sample['label']
        
        num_output_channels = 1 if image.mode == 'L' else 3
        
        if random.random() < self.p:
            
            if task_name in ['jigsaw']:
                return {'image': F.to_grayscale(image, num_output_channels=num_output_channels), 'label': label}
            
            else:
                raise ValueError(f'task name {task_name} not available!')
        else:
            return sample
        
class ColorJitter(T.ColorJitter):
    
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)
    
        
    def __call__(self, sample, task_name):
        
        t = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        def forward(self, img):
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)
            

            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    img = F.adjust_brightness(img, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    img = F.adjust_contrast(img, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    img = F.adjust_saturation(img, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    img = F.adjust_hue(img, hue_factor)

            return img

        image, label = sample['image'], sample['label'] 
        if task_name in ['autoencoder']:
            return {'image': self.forward(image), 'label': self.forward(label)}
        elif task_name in ['segmentsemantic']:
            return {'image': self.forward(image), 'label': label}
        elif task_name in ['class_scene', 'class_object', 'jigsaw', 'room_layout', 'normal', 'room_layout']:
            return {'image': self.forward(image), 'label': label}
        else:
            raise ValueError(f'task name {task_name} not available!')


def load_class_object_logits(label_path, selected=False, normalize=True, final5k=False):
    try:
        logits = np.load(label_path)
    except:
        print(f'corrupted: {label_path}')
        raise
    lib_data_dir = os.path.abspath(os.path.dirname(__file__))
    if selected:
        selection_file = os.path.join(lib_data_dir, 'class_object_final5k.npy') if final5k else os.path.join(lib_data_dir, 'class_object_selected.npy')
        selection = np.load(selection_file)
        logits = logits[selection.astype(bool)]
        if normalize:
            logits = logits / logits.sum()
    target = np.asarray(logits)
    return target

def load_class_object_label(label_path, selected=False, final5k=False):
    logits = load_class_object_logits(label_path, selected=selected, normalize=True, final5k=final5k)
    target = np.asarray(logits.argmax())
    return target


def load_class_scene_logits(label_path, selected=False, normalize=True, final5k=False):
    try:
        logits = np.load(label_path)
    except:
        raise FileNotFoundError(f'corrupted: {label_path}')
    lib_data_dir = os.path.abspath(os.path.dirname(__file__))
    if selected:
        selection_file = os.path.join(lib_data_dir, 'class_scene_final5k.npy') if final5k else os.path.join(lib_data_dir, 'class_scene_selected.npy')
        selection = np.load(selection_file)
        logits = logits[selection.astype(bool)]
        if normalize:
            logits = logits / logits.sum()
    target = np.asarray(logits)
    return target


def image2tiles(img, permutation, new_dims):
    
    if len(permutation) != 9:
        raise ValueError (f'Target permutation of Jigsaw is supposed to have length 9, getting {len(permutation)} here')
        
    He, Wi = img.size 
    
    unitH = int(He / 3)
    unitW = int(Wi / 3)
    
    endH = new_dims[0]
    endW = new_dims[1]
    
    img_tiles = []
    
    for n in range(9):
        pos = permutation[n]
        posH = int(pos // 3) * unitH
        posW = int(pos % 3) * unitW
        pos_i = torch.randint(0, unitH - endH + 1, size=(1,)).item()
        pos_j = torch.randint(0, unitW - endW + 1, size=(1,)).item()
        img_tiles.append(F.crop(img, posH + pos_i, posW + pos_j, endH, endW))
        
    return img_tiles
    
            
class MakeJigsawPuzzle(object):
    def __init__(self, classes, mode, tile_dim=(64, 64), centercrop=None, norm=True, totensor=True):
        self.classes = classes
        self.mode = mode
        self.permutation_set = get_permutation_set(classes=classes, mode=mode)
        self.tile_dim = tile_dim
        self.centercrop = centercrop
        self.image_norm = norm
        self.totensor = totensor
        
    def __call__(self, sample, task_name):
        assert task_name == 'jigsaw'
        image, permutation_idx = sample['image'], sample['label']
        permutation = self.permutation_set[permutation_idx]
        if task_name == 'jigsaw':
            He, Wi = image.size
            if self.centercrop:
                image = F.center_crop(image, (He * self.centercrop, Wi * self.centercrop))
                image = F.resize(image, (He, Wi), Image.BILINEAR)
            pil_image_tiles = image2tiles(image, permutation, self.tile_dim)
            if self.totensor:
                raws = torch.empty(9, 3, self.tile_dim[0], self.tile_dim[1], dtype=torch.float32)
                images = torch.empty(9, 3, self.tile_dim[0], self.tile_dim[1], dtype=torch.float32)
                for i, tile in enumerate(pil_image_tiles):
                    tile_tensor = F.to_tensor(tile)
                    raws[i, :, :, :] = tile_tensor
                    if self.image_norm:
                        mean = tile_tensor.mean((-2, -1))
                        std = tile_tensor.std((-2, -1)) + 0.0001
                        tile_tensor = F.normalize(tile_tensor, mean, std)
                    images[i, :, :, :] = tile_tensor
            else:
                raws = pil_image_tiles
                images = pil_image_tiles
                
            return {'raw': raws, 'image': images,'label': permutation_idx}
        else:
            TypeError('Only jigsaw can use MakeJigsawPuzzle')
             
    
    
def random_jigsaw_permutation(label_path, classes=1000):
    rand_int = random.randint(0, classes - 1)
    return rand_int


def load_raw_img_label(label_path):
    try:
        image = io.imread(label_path)
    except:
        raise Exception(f'corrupted {label_path}!')
    return image
    

def get_permutation_set(mode, classes=1000):
    assert mode in ['max', 'avg']
    permutation_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), f'permutations_hamming_{mode}_{classes}.npy')
    return np.load(permutation_path)