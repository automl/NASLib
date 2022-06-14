import json
import os
import pickle

from naslib.utils.utils import get_project_root
from naslib.utils.load_ops import TASK_NAMES

"""
This file loads any dataset files or api's needed by the Trainer or ZeroCostPredictorEvaluator object.
They must be loaded outside of the search space object, because search spaces are copied many times
throughout the discrete NAS algos, which would lead to memory errors.
"""

def get_transbench101_api(dataset):
    """
    Load the TransNAS-Bench-101 data
    """
    if dataset not in TASK_NAMES:
        return None

    datafile_path = os.path.join(get_project_root(), "data", "transnas-bench_v10141024.pth")
    assert os.path.exists(datafile_path), f"Could not fine {datafile_path}. Please download transnas-bench_v10141024.pth\
 from https://www.noahlab.com.hk/opensource/vega/page/doc.html?path=datasets/transnasbench101"

    from naslib.search_spaces import TransNASBenchAPI
    api = TransNASBenchAPI(datafile_path)
    return {'api': api, 'task': dataset}


def get_nasbench101_api(dataset=None):
    import naslib.utils.nb101_api as api

    nb101_datapath = os.path.join(get_project_root(), "data", "nasbench_only108.pkl")
    assert os.path.exists(nb101_datapath), f"Could not find {nb101_datapath}. Please download nasbench_only108.pk \
from https://drive.google.com/drive/folders/1rwmkqyij3I24zn5GSO6fGv2mzdEfPIEa"

    nb101_data = api.NASBench(nb101_datapath)
    return {"api": api, "nb101_data": nb101_data}

def get_nasbench201_api(dataset):
    """
    Load the NAS-Bench-201 data
    """
    datafiles = {
        'cifar10': 'nb201_cifar10_full_training.pickle',
        'cifar100': 'nb201_cifar100_full_training.pickle',
        'ImageNet16-120': 'nb201_ImageNet16_full_training.pickle'
    }

    if dataset not in datafiles.keys():
        return None

    datafile_path = os.path.join(get_project_root(), 'data', datafiles[dataset])
    assert os.path.exists(datafile_path), f'Could not find {datafile_path}. Please download {datafiles[dataset]} from \
https://drive.google.com/drive/folders/1rwmkqyij3I24zn5GSO6fGv2mzdEfPIEa'

    with open(datafile_path, 'rb') as f:
        data = pickle.load(f)

    return {"nb201_data": data}


def get_nasbench301_api(dataset):
    if dataset != 'cifar10':
        return None
    # Load the nb301 performance and runtime models
    try:
        import nasbench301
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError('No module named \'nasbench301\'. \
            Please install nasbench301 from https://github.com/automl/nasbench301@no_gin')

    # Paths to v1.0 model files and data file.
    download_path = os.path.join(get_project_root(), "data")
    nb_models_path = os.path.join(download_path, "nb_models_1.0")
    os.makedirs(download_path, exist_ok=True)

    nb301_model_path=os.path.join(nb_models_path, "xgb_v1.0")
    nb301_runtime_path=os.path.join(nb_models_path, "lgb_runtime_v1.0")

    if not all(os.path.exists(model) for model in [nb301_model_path,
                                                   nb301_runtime_path]):
        nasbench301.download_models(version='1.0', delete_zip=True,
                                    download_dir=download_path)

    models_not_found_msg = "Please download v1.0 models from \
https://figshare.com/articles/software/nasbench301_models_v1_0_zip/13061510"

    # Verify the model and data files exist
    assert os.path.exists(nb_models_path), f"Could not find {nb_models_path}. {models_not_found_msg}"
    assert os.path.exists(nb301_model_path), f"Could not find {nb301_model_path}. {models_not_found_msg}"
    assert os.path.exists(nb301_runtime_path), f"Could not find {nb301_runtime_path}. {models_not_found_msg}"

    performance_model = nasbench301.load_ensemble(nb301_model_path)
    runtime_model = nasbench301.load_ensemble(nb301_runtime_path)

    nb301_model = [performance_model, runtime_model]

    return {
        "nb301_model": nb301_model,
    }


def get_dataset_api(search_space=None, dataset=None):

    if search_space == "nasbench101":
        return get_nasbench101_api(dataset=dataset)

    elif search_space == "nasbench201":
        return get_nasbench201_api(dataset=dataset)

    elif search_space == "nasbench301":
        return get_nasbench301_api(dataset=dataset)

    elif search_space in ['transbench101', 'transbench101_micro', 'transbench101_macro']:
        return get_transbench101_api(dataset=dataset)

    elif search_space == "test":
        return None

    else:
        raise NotImplementedError()


def get_zc_benchmark_api(search_space, dataset):

    datafile_path = os.path.join(get_project_root(), "data", f"zc_{search_space}.json")
    with open(datafile_path) as f:
        data = json.load(f)

    return data[dataset]


def load_sampled_architectures(search_space, postfix=''):
    datafile_path = os.path.join(get_project_root(), "data", "archs", f"archs_{search_space}{postfix}.json")
    with open(datafile_path) as f:
        data = json.load(f)

    return data
