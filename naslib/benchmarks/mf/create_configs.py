import argparse
import os
import random
from venv import create
import yaml
from glob import glob
import numpy as np
from addict import Dict
import itertools

# pycodestyle --max-line-length=120 naslib/benchmarks/mf/create_configs.py

# TODO: add max_budget
# TODO: add fidelity
# TODO: add budget

def create_configs(
    start_seed: int = 0,
    trials: int = 100,
    optimizer: str = "rs",
    predictor: str = "xgb",
    fidelity: int = 200,
    acq_fn_optimization: str = "mutation",
    dataset: str = "cifar10",
    out_dir: str = "run",
    checkpoint_freq: int = 5000,
    epochs: int = 150,
    budget: int = 1000,
    search_space: str = "nasbench201",
    HPO: bool = False, 
    num_config: int = 100,
    **kwargs
):
    """Function creates config for given parameters

    Args:
        start_seed: starting seed.
        trials: Number of trials.
        optimizer: which optimizer.
        predictor: which predictor.
        fidelity: Fidelity.
        acq_fn_optimization: acq_fn.
        dataset: Which dataset.
        epochs: How many search epochs.
        search_space: nasbench201 or darts?.
        HPO: Hyperparameter Optimisation enabled/disabled.
        num_config: Number of configs explored by HPO (Random Search).

    Returns:
        No return.

    """
    start_seed = int(start_seed)
    trials = int(trials)
    
    # first generate the default config at config 0
    config_id = 0
    # TODO: add config out dir to config
    folder = f"./configs/{search_space}/{dataset}/{optimizer}/config_{config_id}"
    filename = f"./configs/{search_space}_{dataset}_{optimizer}_config_{config_id}"
    # folder = f"naslib/benchmarks/bbo/configs_cpu/{search_space}/{dataset}/{optimizer}/config_{config_id}"
    os.makedirs(folder, exist_ok=True)       
        
    for seed in range(start_seed, start_seed + trials):
        # np.random.seed(seed)
        # random.seed(seed)

        config = {
            "seed": seed,
            "search_space": search_space,
            "dataset": dataset,
            "optimizer": optimizer,
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
                "max_mutations": 1,
                "num_candidates": 50,
                "checkpoint_freq": checkpoint_freq,
                "epochs": epochs,
                "fidelity": fidelity,
                "min_fidelity": 1,
                "number_archs": 128,
                "budget_type": "epoch",
                "budget_max": 128,
                "method": "random",
                "eta": 2,
                "max_budget": 200,
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
                "max_budget": 200,
                "fidelity": 200,
                "n_process": 1_000_000,
                "budgets": 360_000, 
                # config section for bohb
                "min_bandwith": 0.001,
                "top_n_percent": 0.1,
                "min_points_in_model": 7,
                # config section for dehb
                "enc_dim": 6,
                "max_mutations": 1,
                "crossover_prob": 0.5,
                "mutate_prob": 0.5,
            },
        }
        # path = os.path.join(filename, f"seed_{seed}.yaml")
        path = filename + f"_{seed}.yaml"

        with open(path, "w") as fh:
            yaml.dump(config, fh)
    num_config = num_config if HPO else 1
    for config_id in range(1, num_config):
        folder = f"./configs/{search_space}/{dataset}/{optimizer}/config_{config_id}"
        filename = f"./configs/{search_space}_{dataset}_{optimizer}_config_{config_id}"

        os.makedirs(folder, exist_ok=True)

        # SH/HB
        max_fidelity = 200 # int(np.random.choice(range(100, 200)))
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
            # np.random.seed(seed)
            # random.seed(seed)
            # TODO: max_fidelity should be dependent on eta and min_fidelity
            # fidelity_range = [2**i for i in range(0, 9)]
            # max_fidelity = int(np.random.choice(fidelity_range))
            # min fidelity has to be lower/equal to max_fidelity
            # min_fidelity = int(np.random.choice(
            #     list(
            #         filter(
            #             lambda n: n <= max_fidelity, fidelity_range))))
            config = {
                "seed": seed,
                "search_space": search_space,
                "dataset": dataset,
                "optimizer": optimizer,
                "out_dir": out_dir,
                "config_id": config_id,
                "search": {
                    "checkpoint_freq": checkpoint_freq,
                    "epochs": epochs,
                    "fidelity": fidelity,
                    "sample_size": int(np.random.choice(range(5, 100))),
                    "population_size": int(np.random.choice(range(5, 100))),
                    "num_init": int(np.random.choice(range(5, 100))),
                    "k":int(np.random.choice(range(10, 50))),
                    "num_ensemble": 3,
                    "acq_fn_type": "its",
                    "acq_fn_optimization": acq_fn_optimization,
                    "encoding_type": "path",
                    "num_arches_to_mutate": int(np.random.choice(range(1, 20))),
                    "max_mutations": int(np.random.choice(range(1, 20))),
                    "num_candidates": int(np.random.choice(range(5, 50))),
                    "predictor": predictor,
                    "debug_predictor": False,
                    # config section for successive halving,
                    # config secton for Hyperband,
                    "min_budget": min_budget,
                    "max_budget": max_fidelity,
                    "fidelity": 200,
                    "n_process": 1_000_000,
                    "budgets": 360_000, 
                    "eta": eta,
                    "epsilon": 1e-6,
                    # config section for BOHB
                    "min_bandwith": min_bandwith,
                    "top_n_percent": top_n_percent,
                    "min_points_in_model": 7,
                    # config section for DEHB
                    # config section for dehb
                    "enc_dim": 6,
                    "max_mutations": max_mutations,
                    "crossover_prob": crossover_prob,
                    "mutate_prob": mutate_prob,
                },
            }
            print(f"folder: {folder}")
            # path = os.path.join(filename, f"seed_{seed}.yaml")
            path = filename + f"_{seed}.yaml"
            print(f"path: {path}")
            with open(path, "w") as fh:
                yaml.dump(config, fh)

if __name__ == "__main__":
    main()

def check_config(search_space, dataset, optimizer, trials, HPO, num_config):
    folder = os.path.join(".", "configs", search_space, dataset, optimizer)
    test = os.path.join(folder, "**", "*.yaml")  
    files = glob(test, recursive=True)
    print(f"{len(files)} files")
    num_config = num_config if HPO else 1
    if len(files) == trials * num_config:
        print("Created config file(s) successfully")
        return
    print("Config file(s) not successfully created")
    exit(1)

def main():
    with open("config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
    # convert such that config elements are accessiable via attributes
    config = Dict(config)
    start_seed = config.start_seed if config.start_seed else 0
    trials = config.trials
    end_seed = start_seed + trials - 1

    optimizers = config.optimizers
    # TODO: Implement check for optimizers
    # See lines 13 - 18 in make_configs_nb201.sh

    out_dir = config.out_dir

    config_type = config.config_type
    if config_type not in {'bbo-bs', 'predictor-bs'}:
        print('Invalid config')
        print('config_type either bbo-bs or predictor-bs')
        exit(1)

    search_space = config.search_space

    datasets = config.datasets
    fidelity = config.fidelity
    epochs = config.epochs
    budget = config.budget
    predictor = config.predictor

    HPO = config.HPO
    num_config = config.num_config

    for dataset, optimizer in itertools.product(datasets, optimizers):
        print(f"Creating config for dataset: {dataset}: & optimizer: {optimizer}")
        create_configs(
            start_seed=start_seed,
            trials=trials,
            out_dir=out_dir,
            dataset=dataset,
            config_type=config_type,
            search_space=search_space,
            optimizer=optimizer,
            predictor=predictor,
            fidelity=fidelity,
            epochs=epochs,
            budget=budget,
            HPO=HPO,
            num_config=num_config
        )
        check_config(search_space, dataset, optimizer, trials, HPO, num_config)