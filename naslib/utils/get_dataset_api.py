import os
import pickle

from naslib.utils.utils import get_project_root

"""
This file loads any dataset files or api's needed by the Trainer or PredictorEvaluator object.
They must be loaded outside of the search space object, because search spaces are copied many times
throughout the discrete NAS algos, which would lead to memory errors.
"""

def get_nasbench101_api(dataset=None):
    # load nasbench101
    from nasbench import api
    nb101_data = api.NASBench(os.path.join(get_project_root(), 'data', 'nasbench_only108.tfrecord'))
    return {'api': api, 'nb101_data':nb101_data}


def get_nasbench201_api(dataset=None):
    """
    Load the NAS-Bench-201 data
    """
    if dataset == 'cifar10':
        with open(os.path.join(get_project_root(), 'data', 'nb201_cifar10_full_training.pickle'), 'rb') as f:
            data = pickle.load(f)

    elif dataset == 'cifar100':
        with open(os.path.join(get_project_root(), 'data', 'nb201_cifar100_full_training.pickle'), 'rb') as f:
            data = pickle.load(f)

    elif dataset == 'ImageNet16-120':
        with open(os.path.join(get_project_root(), 'data', 'nb201_ImageNet16_full_training.pickle'), 'rb') as f:
            data = pickle.load(f)

    return {'nb201_data':data}


def get_darts_api(dataset=None):
    """
    Load the nb301 training data (which contains full learning curves) and the nb301 models
    """
    import nasbench301
    data_folder = os.path.join(get_project_root(), 'data/')
    with open(os.path.join(data_folder, 'nb301_full_training.pickle'), 'rb') as f:
        nb301_data = pickle.load(f)
        nb301_arches = list(nb301_data.keys())

    performance_model = nasbench301.load_ensemble(os.path.join(data_folder + 'nb301_models/xgb_v1.0'))
    runtime_model = nasbench301.load_ensemble(os.path.join(data_folder + 'nb301_models/lgb_runtime_v1.0'))
    nb301_model = [performance_model, runtime_model]
    return {'nb301_data': nb301_data, 'nb301_arches':nb301_arches, 'nb301_model':nb301_model}


def get_nlp_api(dataset=None):
    """
    Load the NAS-Bench-NLP data
    """
    with open(os.path.join(get_project_root(), 'data', 'nb_nlp.pickle'), 'rb') as f:
        nlp_data = pickle.load(f)
    nlp_arches = list(nlp_data.keys())
    return {'nlp_data':nlp_data, 'nlp_arches':nlp_arches}


def get_dataset_api(search_space=None, dataset=None):

    if search_space == 'nasbench101':
        return get_nasbench101_api(dataset=dataset)

    elif search_space == 'nasbench201':
        return get_nasbench201_api(dataset=dataset)

    elif search_space == 'darts':
        return get_darts_api(dataset=dataset)
    
    elif search_space == 'nlp':
        return get_nlp_api(dataset=dataset)
    
    else:
        raise NotImplementedError()
