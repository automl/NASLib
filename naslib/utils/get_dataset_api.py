import os
import pickle

from naslib.utils.utils import get_project_root
from naslib.utils.utils_asr import from_folder

"""
This file loads any dataset files or api's needed by the Trainer or PredictorEvaluator object.
They must be loaded outside of the search space object, because search spaces are copied many times
throughout the discrete NAS algos, which would lead to memory errors.
"""

def get_transbench101_api(dataset=None):
    datafile_path = os.path.join(get_project_root(), "data", "transnas-bench_v10141024.pth")
    assert os.path.exists(datafile_path), f"Could not fine {datafile_path}. Please download transnas-bench_v10141024.pth\
 from https://www.noahlab.com.hk/opensource/vega/page/doc.html?path=datasets/transnasbench101"

    from naslib.search_spaces import TransNASBenchAPI
    api = TransNASBenchAPI(datafile_path)
    return {'api': api, 'task': dataset}


def get_nasbench101_api(dataset=None):
    # load nasbench101
    import naslib.utils.nb101_api as api

    nb101_datapath = os.path.join(get_project_root(), "data", "nasbench_only108.pkl")
    assert os.path.exists(nb101_datapath), f"Could not find {nb101_datapath}. Please download nasbench_only108.pk \
from https://drive.google.com/drive/folders/1rwmkqyij3I24zn5GSO6fGv2mzdEfPIEa"

    nb101_data = api.NASBench(nb101_datapath)
    return {"api": api, "nb101_data": nb101_data}


def get_nasbench201_api(dataset=None):
    """
    Load the NAS-Bench-201 data
    """
    datafiles = {
        'cifar10': 'nb201_cifar10_full_training.pickle',
        'cifar100': 'nb201_cifar100_full_training.pickle',
        'ImageNet16-120': 'nb201_ImageNet16_full_training.pickle'
    }

    datafile_path = os.path.join(get_project_root(), 'data', datafiles[dataset])
    assert os.path.exists(datafile_path), f'Could not find {datafile_path}. Please download {datafiles[dataset]} from \
https://drive.google.com/drive/folders/1rwmkqyij3I24zn5GSO6fGv2mzdEfPIEa'

    with open(datafile_path, 'rb') as f:
        data = pickle.load(f)

    return {"nb201_data": data}


def get_darts_api(dataset=None):
    # Paths to v1.0 model files and data file.
    nb_models_path = os.path.join(get_project_root(), "data", "nb_models")
    nb301_model_path=os.path.join(nb_models_path, "xgb_v1.0")
    nb301_runtime_path=os.path.join(nb_models_path, "lgb_runtime_v1.0")
    data_path = os.path.join(get_project_root(), "data", "nb301_full_training.pickle")

    models_not_found_msg = "Please download v1.0 models from \
https://figshare.com/articles/software/nasbench301_models_v1_0_zip/13061510"

    # Verify the model and data files exist
    assert os.path.exists(nb_models_path), f"Could not find {nb_models_path}. {models_not_found_msg}"
    assert os.path.exists(nb301_model_path), f"Could not find {nb301_model_path}. {models_not_found_msg}"
    assert os.path.exists(nb301_runtime_path), f"Could not find {nb301_runtime_path}. {models_not_found_msg}"
    assert os.path.isfile(data_path), f"Could not find {data_path}. Please download nb301_full_training.pickle from\
        https://drive.google.com/drive/folders/1rwmkqyij3I24zn5GSO6fGv2mzdEfPIEa?usp=sharing"

    # Load the nb301 performance and runtime models
    try:
        import nasbench301
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError('No module named \'nasbench301\'. \
            Please install nasbench301 from https://github.com/crwhite14/nasbench301')

    performance_model = nasbench301.load_ensemble(nb301_model_path)
    runtime_model = nasbench301.load_ensemble(nb301_runtime_path)

    with open(data_path, "rb") as f:
        nb301_data = pickle.load(f)
        nb301_arches = list(nb301_data.keys())

    nb301_model = [performance_model, runtime_model]

    return {
        "nb301_data": nb301_data,
        "nb301_arches": nb301_arches,
        "nb301_model": nb301_model,
    }


def get_dataset_api(search_space=None, dataset=None):

    if search_space == "nasbench101":
        return get_nasbench101_api(dataset=dataset)

    elif search_space == "nasbench201":
        return get_nasbench201_api(dataset=dataset)

    elif search_space == "darts":
        return get_darts_api(dataset=dataset)

    elif search_space in ['transbench101', 'transbench101_micro', 'transbench101_macro']:
        return get_transbench101_api(dataset=dataset)

    elif search_space == "test":
        return None

    else:
        raise NotImplementedError()

