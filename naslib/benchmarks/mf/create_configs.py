import os


import yaml
from glob import glob
import numpy as np
from addict import Dict
import itertools

def dump_configs(
    configs: list,
    optimizer: str,
    dataset: str,
    search_space: str,
    out_dir = 'run'
):
    # setting default values specific to search space/dataset combination
    if search_space == 'nasbench201':
        fidelity = 200
        max_budget = fidelity
        enc_dim = 6
        min_points_in_model = enc_dim + 1

        if dataset == 'cifar10':
            budgets = 1_100_000
        elif dataset == 'cifar100':
            budgets = 2_000_000
        elif dataset == 'ImageNet16-120':
            budgets = 6_500_5000
        else:
            return
        
    elif search_space == 'nasbench311':
        fidelity = 97
        max_budget = fidelity
        enc_dim = 32
        min_points_in_model = enc_dim + 1
        if dataset == 'cifar10':
            budgets = 2_500_000
        else: 
            return

    elif search_space == 'asr':
        fidelity = 39
        max_budget = fidelity
        enc_dim = 6
        min_points_in_model = enc_dim + 1
        if dataset != 'TIMIT':
            return
        budgets = 16_000
    else:
        return

    os.makedirs(out_dir, exist_ok=True)
    for config in configs:
        config_id = config['config_id']
        seed = config['seed']
        filename = os.path.join(".", out_dir, f"{search_space}_{dataset}_{optimizer}_config_{config_id}_{seed}.yaml")

        with open(filename, "w") as fh:
            config['search_space'] = search_space
            config['dataset'] = dataset
            config['optimizer'] = optimizer

            config['search']['max_budget'] = max_budget
            config['search']['fidelity'] = fidelity
            config['search']['budgets'] = budgets
            config['search']['min_points_in_model'] = min_points_in_model
            config['search']['enc_dim'] = enc_dim
            yaml.dump(config, fh)

def create_configs(
    start_seed: int = 0,
    trials: int = 100,
    predictor: str = "xgb",
    fidelity: int = 200,
    acq_fn_optimization: str = "mutation",
    out_dir: str = "run",
    checkpoint_freq: int = 5000,
    epochs: int = 400,
    num_config: int = 100,
    **kwargs
):
    """Function creates config for given parameters

    Args:
        start_seed: starting seed.
        trials: Number of trials.
        predictor: which predictor.
        fidelity: Fidelity.
        acq_fn_optimization: acq_fn.
        epochs: How many search epochs.
        num_config: Number of configs explored by HPO (Random Search).

    Returns:
        configs.

    """
    start_seed = int(start_seed)
    trials = int(trials)
    # first generate the default config at config 0
    config_id = 0
    configs = []
    for seed in range(start_seed, start_seed + trials):
        config = {
            "seed": seed,
            "search_space": None,
            "dataset": None,
            "optimizer": None,
            "out_dir": out_dir,
            "config_id": config_id,
            "search": {
                "sample_size": 10,
                "population_size": 50,
                "num_init": 10,
                "k":10,
                "num_ensemble": 3,
                "acq_fn_type": "its",
                "num_arches_to_mutate": 1,
                "num_candidates": 50,
                "checkpoint_freq": checkpoint_freq,
                "epochs": epochs,
                "number_archs": 128,
                "method": "random",
                "eta": 2,
                "max_budget": None,
                "min_budget": 1,
                "n_process": 1000,
                "epsilon": 1e-6,
                "num_ensemble": 3,
                "acq_fn_type": "its",
                "acq_fn_optimization": acq_fn_optimization,
                "encoding_type": "path",
                "predictor": predictor,
                "debug_predictor": False,
                # config secton for successive halving/ hyperband,
                "min_budget": 1,
                "fidelity": fidelity,
                "n_process": 1_000_000,
                "budgets": None, 
                # config section for bohb
                "min_bandwith": 0.001,
                "top_n_percent": 0.1,
                "min_points_in_model": None,
                # config section for dehb
                "enc_dim": None,
                "max_mutations": 1,
                "crossover_prob": 0.5,
                "mutate_prob": 0.5,
            },
        }
        configs.append(config)
        
    for config_id in range(1, num_config):
        sample_size = int(np.random.choice(range(5, 100)))
        population_size = int(np.random.choice(range(5, 100)))
        num_init = int(np.random.choice(range(5, 100)))
        k = int(np.random.choice(range(10, 50)))
        num_arches_to_mutate = int(np.random.choice(range(1, 20)))
        max_mutations = int(np.random.choice(range(1, 20)))
        num_candidates = int(np.random.choice(range(5, 50)))

        # SH/HB
        min_budget = int(np.random.choice(range(1, 50)))
        eta = int(np.random.choice(range(2, 5)))
        # BOHB
        min_bandwith = float(np.random.choice(np.arange(0.0, 0.011, 0.001)))
        top_n_percent = float(np.random.choice(np.arange(0.05, 0.31, 0.01)))
        # DEHB
        max_mutations = int(np.random.choice(range(1, 5)))
        crossover_prob = float(np.random.choice(np.arange(0.0, 1.10, 0.1)))
        mutate_prob = float(np.random.choice(np.arange(0.0, 1.10, 0.1)))

        for seed in range(start_seed, start_seed + trials):
            config = {
                "seed": seed,
                "search_space": None,
                "dataset": None,
                "optimizer": None,
                "out_dir": out_dir,
                "config_id": config_id,
                "search": {
                    "checkpoint_freq": checkpoint_freq,
                    "epochs": epochs,
                    "fidelity": fidelity,
                    "sample_size": sample_size,
                    "population_size": population_size,
                    "num_init": num_init,
                    "k": k,
                    "num_ensemble": 3,
                    "acq_fn_type": "its",
                    "acq_fn_optimization": acq_fn_optimization,
                    "encoding_type": "path",
                    "num_arches_to_mutate": num_arches_to_mutate,
                    "max_mutations": max_mutations,
                    "num_candidates": num_candidates,
                    "predictor": predictor,
                    "debug_predictor": False,
                    # config section for successive halving,
                    # config secton for Hyperband,
                    "min_budget": min_budget,
                    "max_budget": None,
                    "fidelity": fidelity,
                    "n_process": 1_000_000,
                    "budgets": None, 
                    "eta": eta,
                    "epsilon": 1e-6,
                    # config section for BOHB
                    "min_bandwith": min_bandwith,
                    "top_n_percent": top_n_percent,
                    "min_points_in_model": None,
                    # config section for DEHB
                    # config section for dehb
                    "enc_dim": None,
                    "max_mutations": max_mutations,
                    "crossover_prob": crossover_prob,
                    "mutate_prob": mutate_prob,
                },
            }
            configs.append(config)
    return configs


def check_config(out_dir):
    folder = os.path.join(out_dir)
    config_files_path = os.path.join(folder, "**", "*.yaml")  
    files = glob(config_files_path, recursive=True)
    print(f"{len(files)} config file(s) created")

def main():
    with open("/Users/lars/Projects/NASLib_cleanup/naslib/benchmarks/mf/config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
    # convert such that config elements are accessiable via attributes
    config = Dict(config)
    start_seed = config.start_seed if config.start_seed else 0
    trials = config.trials

    optimizers = config.optimizers

    out_dir = config.out_dir
    out_dir_configs = config.out_dir_configs

    search_spaces = config.search_space
    datasets = config.datasets
    num_config = config.num_config
    configs = create_configs(
            start_seed=start_seed,
            trials=trials,
            out_dir=out_dir,
            num_config=num_config
        )
    for search_space, dataset, optimizer in itertools.product(search_spaces, datasets, optimizers):        
        dump_configs(dataset=dataset, optimizer=optimizer, search_space=search_space, configs=configs, out_dir=out_dir_configs)
    check_config(out_dir_configs)

if __name__ == "__main__":
    main()