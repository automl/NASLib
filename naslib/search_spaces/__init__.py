from .nasbench101.graph import NasBench101SearchSpace
from .nasbench201.graph import NasBench201SearchSpace
from .nasbench301.graph import NasBench301SearchSpace
from .transbench101.graph import TransBench101SearchSpaceMicro
from .transbench101.graph import TransBench101SearchSpaceMacro

from .transbench101.api import TransNASBenchAPI

supported_search_spaces = {
    "nasbench101": NasBench101SearchSpace,
    "nasbench201": NasBench201SearchSpace,
    "nasbench301": NasBench301SearchSpace,
    'transbench101_micro': TransBench101SearchSpaceMicro,
    'transbench101_macro': TransBench101SearchSpaceMacro,
}

dataset_n_classes = {
    "cifar10": 10,
    "cifar100": 100,
    "imagenet16-120": 120,
}

def get_search_space(name, dataset):
    search_space_cls = supported_search_spaces[name.lower()]

    if name == 'transbench101_micro' or name == 'transbench101_macro':
        return search_space_cls(dataset=dataset)

    n_classes = dataset_n_classes[dataset.lower()]
    return search_space_cls(n_classes=n_classes)
    
