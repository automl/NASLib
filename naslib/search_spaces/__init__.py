from .darts.graph import DartsSearchSpace
from .nasbench101.graph import NasBench101SearchSpace
from .nasbench201.graph import NasBench201SearchSpace
from .nasbenchnlp.graph import NasBenchNLPSearchSpace
from .nasbenchasr.graph import NasBenchASRSearchSpace
from .natsbenchsize.graph import NATSBenchSizeSearchSpace
from .transbench101.graph import TransBench101SearchSpaceMicro
from .transbench101.graph import TransBench101SearchSpaceMacro

from .transbench101.api import TransNASBenchAPI


supported_search_spaces = {
    "darts": DartsSearchSpace,
    "nasbench101": NasBench101SearchSpace,
    "nasbench201": NasBench201SearchSpace,
    "nlp": NasBenchNLPSearchSpace,
    "asr": NasBenchASRSearchSpace,
    "nastbenchsize": NATSBenchSizeSearchSpace,
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
    n_classes = dataset_n_classes[dataset.lower()]

    if name == 'transbench101_micro':
        return search_space_cls(dataset=dataset)

    # TODO_COMPETITION: Change all search space initializers to have this signature
    return search_space_cls(n_classes=n_classes)
    