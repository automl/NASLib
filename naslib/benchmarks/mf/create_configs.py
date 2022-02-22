import argparse
import os
import random
from venv import create
import yaml

import numpy as np


def create_configs(
    config_type: str,
    start_seed: int = 0,
    trials: int = 100,
    optimizer: str = "rs",
    predictor_type: str = "full",
    predictor: str = "xgb",
    test_size: int = 30,
    uniform_random: int = 1, 
    train_size_single:int = 10,
    fidelity_single: int = 5,
    fidelity: int = 200,
    acq_fn_optimization: str = "mutation",
    dataset: str = "cifar10",
    out_dir: str = "run",
    checkpoint_freq: int = 5000,
    epochs: int = 150,
    search_space: str = "nasbench201",
    experiment_type: str = "single",
    run_acc_stats: int = 1,
    max_set_size: int = 10000,
    run_nbhd_size: int = 1,
    max_nbhd_trials: int = 1000,
    run_autocorr: int = 1, 
    max_autocorr_trials: int = 10,
    autocorr_size: int = 36,
    walks: int = 1000,
    HPO: bool = False, 
    num_config: int = 100,
    **kwargs
):
    """Function creates config for given parameters

    Args:
        start_seed: starting seed.
        trials: Number of trials.
        optimizer: which optimizer.
        predictor_type: which predictor.
        predictor: which predictor.
        test_size: Test set size for predictor.
        uniform_random: Train/test set generation type (bool).
        train_size_single: Train size if exp type is single.
        fidelity_single: Fidelity if exp type is single.
        fidelity: Fidelity.
        acq_fn_optimization: acq_fn.
        dataset: Which dataset.
        epochs: How many search epochs.
        config_type: nas or predictor?.
        search_space: nasbench201 or darts?.
        experiment_type: type of experiment.
        run_acc_stats: run accuracy statistics.
        max_set_size: size of val_acc stat computation.
        run_nbhd_size: run experiment to compute nbhd size.
        max_nbhd_trials: size of nbhd size computation. 
        run_autocorr: run experiment to compute autocorrelation.
        max_autocorr_trials: number of autocorrelation trials.
        autocorr_size: size of autocorrelation to test.
        walks: number of random walks.
        HPO: Hyperparameter Optimisation enabled/disabled.
        num_config: Number of configs explored by HPO (Random Search).

    Returns:
        No return.

    """
    if config_type == 'bbo-bs':
        start_seed = int(start_seed)
        trials = int(trials)
        num_config = 100 
        
        # first generate the default config at config 0
        config_id = 0
        # TODO: add config out dir to config
        folder = f"./configs/{search_space}/{dataset}/{optimizer}/config_{config_id}"
        # folder = f"naslib/benchmarks/bbo/configs_cpu/{search_space}/{dataset}/{optimizer}/config_{config_id}"
        os.makedirs(folder, exist_ok=True)       
            
        for seed in range(start_seed, start_seed + trials):
            np.random.seed(seed)
            random.seed(seed)

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
                    "eta": 3,
                    "num_ensemble": 3,
                    "acq_fn_type": "its",
                    "acq_fn_optimization": acq_fn_optimization,
                    "encoding_type": "path",
                    "predictor": predictor,
                    "debug_predictor": False,
                    # config secton for successive halving,
                    "min_budget": 1,
                    "max_budget": 200,
                    "fidelity": 200,
                    "n_process": 1_000_000,
                    "budgets": 360_000, 
                },
            }
            path = os.path.join(folder, f"seed_{seed}.yaml")
            with open(path, "w") as fh:
                yaml.dump(config, fh)
        num_config = num_config if HPO else 1
        for config_id in range(1, num_config):
            folder = f"./configs/{search_space}/{dataset}/{optimizer}/config_{config_id}"
            os.makedirs(folder, exist_ok=True)
            
            for seed in range(start_seed, start_seed + trials):
                np.random.seed(seed)
                random.seed(seed)
                # TODO: max_fidelity should be dependent on eta and min_fidelity
                eta = int(np.random.choice(range(2, 5)))
                fidelity_range = [2**i for i in range(0, 9)]
                max_fidelity = int(np.random.choice(fidelity_range))
                # min fidelity has to be lower/equal to max_fidelity
                min_fidelity = int(np.random.choice(
                    list(
                        filter(
                            lambda n: n <= max_fidelity, fidelity_range))))
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
                        # config secton for successive halving,
                        "min_budget": min_fidelity,
                        "max_budget": max_fidelity,
                        "fidelity": 200,
                        "n_process": 1_000_000,
                        "budgets": 360_000, 
                        "eta": eta,
                        # config section for BOHB
                        "tpe_bandwidth": float(np.random.choice(np.arange(0.01, 1.0, 0.01))), # TODO: what is a good range for tpe??
                    },
                }
                print(f"folder: {folder}")
                path = os.path.join(folder, f"seed_{seed}.yaml")
                print(f"path: {path}")
                with open(path, "w") as fh:
                    yaml.dump(config, fh)
    
    elif config_type == "predictor-bs":
        folder = f"naslib/benchmarks/predictors-bs/configs_{search_space}/{dataset}"
        os.makedirs(folder, exist_ok=True)
        start_seed = int(start_seed)
        trials = int(trials)

        if search_space == "nasbench101":
            total_epochs = 108 - 1
            max_train_size = 1000
        elif search_space == "nasbench201":
            total_epochs = 200 - 1
            max_train_size = 1000
        elif search_space == "darts":
            total_epochs = 96 - 1
            max_train_size = 500
        elif search_space == "nlp":
            total_epochs = 50 - 1
            max_train_size = 1000

        train_size_list = [
            int(j)
            for j in np.logspace(
                start=np.log(5.1) / np.log(2),
                stop=np.log(max_train_size) / np.log(2),
                num=11,
                endpoint=True,
                base=2.0,
            )
        ]
        # train_size_list = [i for i in train_size_list if i < 230]
        fidelity_list = [
            int(j)
            for j in np.logspace(
                start=0.9,
                stop=np.log(total_epochs) / np.log(2),
                num=15,
                endpoint=True,
                base=2.0,
            )
        ]

        if search_space == "nlp":
            fidelity_list.pop(2)
            fidelity_list.insert(5, 6)

        if "svr" in predictor:
            train_size_list.pop(0)
            fidelity_list.pop(0)
            fidelity_list.pop(0)

        for i in range(start_seed, start_seed + trials):
            config = {
                "seed": i,
                "search_space": search_space,
                "dataset": dataset,
                "out_dir": out_dir,
                "predictor": predictor,
                "test_size": test_size,
                "uniform_random": uniform_random,
                "experiment_type": experiment_type,
                "train_size_list": train_size_list,
                "train_size_single": train_size_single,
                "fidelity_single": fidelity_single,
                "fidelity_list": fidelity_list,
                "max_hpo_time": 900,
            }

            with open(folder + f"/config_{predictor}_{i}.yaml", "w") as fh:
                yaml.dump(config, fh)

    elif config_type == "nas":
        folder = f"{out_dir}/{dataset}/configs/nas"
        os.makedirs(folder, exist_ok=True)
        start_seed = int(start_seed)
        trials = int(trials)

        for i in range(start_seed, start_seed + trials):
            config = {
                "seed": i,
                "search_space": search_space,
                "dataset": dataset,
                "optimizer": optimizer,
                "out_dir": out_dir,
                "search": {
                    "checkpoint_freq": checkpoint_freq,
                    "epochs": epochs,
                    "fidelity": 200,
                    "sample_size": 10,
                    "population_size": 30,
                    "num_init": 10,
                    "k": 25,
                    "num_ensemble": 3,
                    "acq_fn_type": "its",
                    "acq_fn_optimization": "mutation",
                    "encoding_type": "path",
                    "num_arches_to_mutate": 2,
                    "max_mutations": 1,
                    "num_candidates": 100,
                    "predictor_type": "feedforward",
                    "debug_predictor": False,
                },
            }

            if optimizer == "lcsvr" and experiment_type == "vary_fidelity":
                path = folder + f"/config_{optimizer}_train_{i}.yaml"
            if optimizer == "lcsvr" and experiment_type == "vary_train_size":
                path = folder + f"/config_{optimizer}_fidelity_{i}.yaml"
            else:
                path = folder + f"/config_{optimizer}_{i}.yaml"

            with open(path, "w") as fh:
                yaml.dump(config, fh)

    elif config_type == "predictor":
        folder = f"{out_dir}/{dataset}/configs/predictors"
        os.makedirs(folder, exist_ok=True)
        start_seed = int(start_seed)
        trials = int(trials)

        if search_space == "nasbench101":
            total_epochs = 108 - 1
            max_train_size = 1000
        elif search_space == "nasbench201":
            total_epochs = 200 - 1
            max_train_size = 1000
        elif search_space == "darts":
            total_epochs = 96 - 1
            max_train_size = 500
        elif search_space == "nlp":
            total_epochs = 50 - 1
            max_train_size = 1000

        train_size_list = [
            int(j)
            for j in np.logspace(
                start=np.log(5.1) / np.log(2),
                stop=np.log(max_train_size) / np.log(2),
                num=11,
                endpoint=True,
                base=2.0,
            )
        ]
        # train_size_list = [i for i in train_size_list if i < 230]
        fidelity_list = [
            int(j)
            for j in np.logspace(
                start=0.9,
                stop=np.log(total_epochs) / np.log(2),
                num=15,
                endpoint=True,
                base=2.0,
            )
        ]

        if search_space == "nlp":
            fidelity_list.pop(2)
            fidelity_list.insert(5, 6)

        if "svr" in predictor:
            train_size_list.pop(0)
            fidelity_list.pop(0)
            fidelity_list.pop(0)

        elif "omni" in predictor and search_space != "darts":
            train_size_list.pop(0)
            train_size_list.pop(-1)
            fidelity_list.pop(1)

        elif "omni" in predictor and search_space == "darts":
            train_size_list.pop(0)
            train_size_list.pop(-1)
            fidelity_list.pop(1)
            fidelity_list.pop(1)

        for i in range(start_seed, start_seed + trials):
            config = {
                "seed": i,
                "search_space": search_space,
                "dataset": dataset,
                "out_dir": out_dir,
                "predictor": predictor,
                "test_size": test_size,
                "uniform_random": uniform_random,
                "experiment_type": experiment_type,
                "train_size_list": train_size_list,
                "train_size_single": train_size_single,
                "fidelity_single": fidelity_single,
                "fidelity_list": fidelity_list,
                "max_hpo_time": 900,
            }

            with open(folder + f"/config_{predictor}_{i}.yaml", "w") as fh:
                yaml.dump(config, fh)

    elif config_type == "nas_predictor":
        folder = f"{out_dir}/{dataset}/configs/nas_predictors"
        os.makedirs(folder, exist_ok=True)
        start_seed = int(start_seed)
        trials = int(trials)

        for i in range(start_seed, start_seed + trials):
            config = {
                "seed": i,
                "search_space": search_space,
                "dataset": dataset,
                "optimizer": optimizer,
                "out_dir": out_dir,
                "search": {
                    "predictor_type": predictor,
                    "epochs": epochs,
                    "checkpoint_freq": checkpoint_freq,
                    "fidelity": 200,
                    "sample_size": 10,
                    "population_size": 30,
                    "num_init": 20,
                    "k": 20,
                    "num_ensemble": 3,
                    "acq_fn_type": "its",
                    "acq_fn_optimization": "random_sampling",
                    "encoding_type": "adjacency_one_hot",
                    "num_arches_to_mutate": 5,
                    "max_mutations": 1,
                    "num_candidates": 200,
                    "batch_size": 256,
                    "data_size": 25000,
                    "cutout": False,
                    "cutout_length": 16,
                    "cutout_prob": 1.0,
                    "train_portion": 0.7,
                },
            }

            path = folder + f"/config_{optimizer}_{predictor}_{i}.yaml"
            with open(path, "w") as fh:
                yaml.dump(config, fh)

    elif config_type == "statistics":
        folder = f"{out_dir}/{search_space}/{dataset}/configs/statistics"
        os.makedirs(folder, exist_ok=True)
        start_seed = int(start_seed)
        trials = int(trials)

        for i in range(start_seed, start_seed + trials):
            config = {
                "seed": i,
                "search_space": search_space,
                "dataset": dataset,
                "out_dir": out_dir,
                "run_acc_stats": run_acc_stats,
                "max_set_size": max_set_size,
                "run_nbhd_size": run_nbhd_size,
                "max_nbhd_trials": max_nbhd_trials,
                "run_autocorr": run_autocorr,
                "max_autocorr_trials": max_autocorr_trials,
                "autocorr_size": autocorr_size,
                "walks": walks,
            }

            with open(folder + f"/config_{i}.yaml", "w") as fh:
                yaml.dump(config, fh)

    else:
        print("invalid config type in create_configs.py")

if __name__ == "__main__":
    """This is executed when run from the command line"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--start_seed", type=int, default=0, help="starting seed")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials")
    parser.add_argument("--optimizer", type=str, default="rs", help="which optimizer")
    parser.add_argument("--predictor_type", type=str, default="full", help="which predictor")
    parser.add_argument("--predictor", type=str, default="xgb", help="which predictor")
    parser.add_argument(
        "--test_size", type=int, default=30, help="Test set size for predictor"
    )
    parser.add_argument(
        "--uniform_random",
        type=int,
        default=1,
        help="Train/test set generation type (bool)",
    )
    parser.add_argument(
        "--train_size_single",
        type=int,
        default=10,
        help="Train size if exp type is single",
    )
    parser.add_argument(
        "--fidelity_single", type=int, default=5, help="Fidelity if exp type is single"
    )
    parser.add_argument(
        "--fidelity", type=int, default=200, help="Fidelity"
    )
    parser.add_argument(
        "--acq_fn_optimization", type=str, default="mutation", help="acq_fn"
    )
    parser.add_argument("--dataset", type=str, default="cifar10", help="Which dataset")
    parser.add_argument("--out_dir", type=str, default="run", help="Output directory")
    parser.add_argument(
        "--checkpoint_freq", type=int, default=5000, help="How often to checkpoint"
    )
    parser.add_argument(
        "--epochs", type=int, default=150, help="How many search epochs"
    )
    parser.add_argument(
        "--config_type", type=str, default="nas", help="nas or predictor?"
    )
    parser.add_argument(
        "--search_space", type=str, default="nasbench201", help="nasbench201 or darts?"
    )
    parser.add_argument(
        "--experiment_type", type=str, default="single", help="type of experiment"
    )
    parser.add_argument(
        "--run_acc_stats", type=int, default=1, help="run accuracy statistics"
    )
    parser.add_argument(
        "--max_set_size", type=int, default=10000, help="size of val_acc stat computation"
    )    
    parser.add_argument(
        "--run_nbhd_size", type=int, default=1, help="run experiment to compute nbhd size"
    )    
    parser.add_argument(
        "--max_nbhd_trials", type=int, default=1000, help="size of nbhd size computation"
    )
    parser.add_argument(
        "--run_autocorr", type=int, default=1, help="run experiment to compute autocorrelation"
    )
    parser.add_argument(
        "--max_autocorr_trials", type=int, default=10, help="number of autocorrelation trials"
    )
    parser.add_argument(
        "--autocorr_size", type=int, default=36, help="size of autocorrelation to test"
    )
    parser.add_argument(
        "--walks", type=int, default=1000, help="number of random walks"
    )
    parser.add_argument(
        "--HPO", type=bool, default=False, help="Optimisation enabled/disabled"
    )
    parser.add_argument(
        "--num_config", type=int, default=100, help="Number of configs explored by HPO (Random Search)"
    )
    
    args = parser.parse_args()
    arguments = vars(args)
    create_configs(**arguments)