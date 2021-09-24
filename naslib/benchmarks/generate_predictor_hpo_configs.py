import argparse
import os
import random
import json
import yaml
import numpy as np

from naslib.predictors.trees.ngb import loguniform


def create_hpo_json_file(hposeed, hpo_config_file, seed=0):

    if hposeed == 0:
        return "None"
    
    # set the seed used to generate hyperparameters, for consistency
    np.random.seed(seed)
    random.seed(seed)

    # generate hpo configs
    hparams_dicts = []
    for i in range(args.num_configs):
        hparams_dict = {
            "bohamiann": {
                "num_steps": int(np.random.choice(range(10, 500))),
            },
            "gp": {
                "num_steps": int(np.random.choice(range(10, 500))),
            },
            "mlp": {
                "num_layers": int(np.random.choice(range(5, 25))),
                "layer_width": int(np.random.choice(range(5, 25))),
                "batch_size": 32,
                "lr": np.random.choice([0.1, 0.01, 0.005, 0.001, 0.0001]),
                "regularization": 0.2,
            },
            "rf": {
                "n_estimators": int(loguniform(16, 128)),
                "max_features": loguniform(0.1, 0.9),
                "min_samples_leaf": int(np.random.choice(19) + 1),
                "min_samples_split": int(np.random.choice(18) + 2),
                "bootstrap": False,
            },
            # note: currently nao and seminas both use params from "seminas"
            "seminas": {
                "gcn_hidden": int(loguniform(16, 128)),
                "batch_size": int(loguniform(32, 256)),
                "lr": loguniform(0.00001, 0.1),
            },
            "xgb": {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "booster": "gbtree",
                "max_depth": int(np.random.choice(range(1, 15))),
                "min_child_weight": int(np.random.choice(range(1, 10))),
                "colsample_bytree": np.random.uniform(0.0, 1.0),
                "learning_rate": loguniform(0.001, 0.5),
                "colsample_bylevel": np.random.uniform(0.0, 1.0),
            }
        }
        hparams_dicts.append(hparams_dict)
        
    hparams_dict = hparams_dicts[hposeed - 1]
    json.dump(hparams_dict, open(hpo_config_file, 'w'))
    print('created', hpo_config_file)


def main(args):

    hpo_config_file = args.hpo_config_folder + "/hpo_" + str(args.hposeed) + ".json"
    if not os.path.exists(hpo_config_file):
        os.makedirs(args.hpo_config_folder, exist_ok=True)
        create_hpo_json_file(args.hposeed, hpo_config_file)

    config = {
        "out_dir": args.out_dir,
        "save": args.save_folder,
        "seed": args.seed,
        "search_space": args.search_space,
        "dataset": args.dataset,
        "predictor": args.predictor,
        "test_size": args.test_size,
        "uniform_random": 1,
        "experiment_type": 'single',
        "train_size_list": [100],
        "train_size_single": args.train_size_single,
        "fidelity_single": 10,
        "fidelity_list": [10],
        "max_hpo_time": 0,
        "hparams_from_file": hpo_config_file,
    }

    file_len = len(args.config_file.split("/")[-1])
    config_path = args.config_file[:-file_len]
    os.makedirs(config_path, exist_ok=True)
    with open(args.config_file, "w") as fh:
        yaml.dump(config, fh)


if __name__ == "__main__":
    """This is executed when run from the command line"""
    parser = argparse.ArgumentParser()

    # experiment configs
    parser.add_argument("--save_folder", type=str, default=None, help="full save folder path")
    parser.add_argument("--config_file", type=str, default=None, help="path to config file")
    parser.add_argument("--out_dir", type=str, default=None, help="out directory")
    parser.add_argument("--search_space", type=str, default=None, help="search space")
    parser.add_argument("--dataset", type=str, default=None, help="dataset")
    parser.add_argument("--predictor", type=str, default=None, help="predictor")
    parser.add_argument("--hposeed", type=int, default=0, help="seed for hyperparameters")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--train_size_single", type=int, default=100, help="train_size")
    parser.add_argument("--test_size", type=int, default=200, help="test size")

    # hpo configs
    parser.add_argument("--hpo_config_folder", type=str, default="predictor_hpo", help="hpo config folder")
    parser.add_argument("--num_configs", type=int, default=1000, help="number of hpo configs")

    args = parser.parse_args()

    main(args)