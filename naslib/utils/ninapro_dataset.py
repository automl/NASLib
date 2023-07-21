import os
import numpy as np
import torch.utils.data
import torchvision.transforms


# adapted from https://github.com/rtu715/NAS-Bench-360/blob/0d1af0ce37b5f656d6491beee724488c3fccf871/perceiver-io/perceiver/data/nb360/ninapro.py#L64
class NinaPro(torch.utils.data.Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.x = np.load(os.path.join(root, f"ninapro_{split}.npy")).astype(np.float32)
        self.x = self.x[:, np.newaxis, :, :].transpose(0, 2, 3, 1)
        self.y = np.load(os.path.join(root, f"label_{split}.npy")).astype(int)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.x[idx, :]
        y = self.y[idx]

        if self.transform:
            x = self.transform(x)
        return x, y


def ninapro_transform(args, channels_last: bool = True):
    transform_list = []

    def channels_to_last(img: torch.Tensor):
        return img.permute(1, 2, 0).contiguous()

    transform_list.append(torchvision.transforms.ToTensor())

    if channels_last:
        transform_list.append(channels_to_last)

    return torchvision.transforms.Compose(transform_list), torchvision.transforms.Compose(transform_list)

