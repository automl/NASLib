import argparse
import os
import random
import json
import numpy as np


def main(args):

    np.random.seed(args.seed)
    random.seed(args.seed)

    # generate json files
    for i in range(args.num_configs):
        hparams_dict = {
            "mlp": {
                "num_layers": int(np.random.choice(range(5, 25))),
                "layer_width": int(np.random.choice(range(5, 25))),
                "batch_size": 32,
                "lr": np.random.choice([0.1, 0.01, 0.005, 0.001, 0.0001]),
                "regularization": 0.2,
            },
            "gp": {
                "num_steps": 200,
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