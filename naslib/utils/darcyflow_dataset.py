import os
import numpy as np
import scipy.io
import torch.utils.data
import torchvision.transforms


class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def __call__(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

# adapted from https://github.com/rtu715/NAS-Bench-360/blob/0d1af0ce37b5f656d6491beee724488c3fccf871/perceiver-io/perceiver/data/nb360/darcyflow.py#L73
def load_darcyflow_data(path):
    train_path = os.path.join(path, "piececonst_r421_N1024_smooth1.mat")
    test_path = os.path.join(path, "piececonst_r421_N1024_smooth2.mat")

    r = 5
    s = int(((421 - 1) / r) + 1)

    x_train, y_train = read_mat(train_path, r, s)
    x_test, y_test = read_mat(test_path, r, s)

    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer(x_train)
    x_test = x_normalizer(x_test)

    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train = y_normalizer(y_train)
    y_test = y_normalizer(y_test)

    x_train = x_train.reshape((-1, s, s, 1))
    x_test = x_test.reshape((-1, s, s, 1))

    trainset = torch.utils.data.TensorDataset(x_train, y_train)
    testset = torch.utils.data.TensorDataset(x_test, y_test)

    return trainset, testset


def read_mat(file_path, r, s):
    data = scipy.io.loadmat(file_path)
    x = read_mat_field(data, "coeff")[:, ::r, ::r][:, :s, :s]
    y = read_mat_field(data, "sol")[:, ::r, ::r][:, :s, :s]
    del data
    return x, y


def read_mat_field(mat, field):
    x = mat[field]
    x = x.astype(np.float32)
    return torch.from_numpy(x)


def darcyflow_transform(args):
    transform_list = []
    transform_list.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(transform_list), torchvision.transforms.Compose(transform_list)
