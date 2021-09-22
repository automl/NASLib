import argparse
import os
import random
import json
import numpy as np

from naslib.predictors.trees.ngb import loguniform


def main(args):

    # set the seed for consistency
    np.random.seed(args.seed)
    random.seed(args.seed)

    # generate json files
    for i in range(args.num_configs):
        hparams_dict = {
            "gp": {
                "num_steps": 200,
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
        file = args.save_folder + "/hpo_config_{}.json".format(i)
        json.dump(hparams_dict, open(file, 'w'))

if __name__ == "__main__":
    """This is executed when run from the command line"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_configs", type=int, default=10, help="number of random configs to generate")
    parser.add_argument("--save_folder", type=str, default="predictor_hpo_configs", help="name of folder to save results")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    
    args = parser.parse_args()

    main(args)