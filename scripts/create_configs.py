import argparse
import os
import random
import yaml

import numpy as np


def main(args):
    folder = f"{args.out_dir}/{args.dataset}/configs/predictors"
    os.makedirs(folder, exist_ok=True)
    args.start_seed = int(args.start_seed)
    args.trials = int(args.trials)

    for i in range(args.start_seed, args.start_seed + args.trials):
        config = {
            "seed": i,
            "search_space": args.search_space,
            "dataset": args.dataset,
            "out_dir": args.out_dir,
            "predictor": args.predictor,
            "test_size": args.test_size,
            "train_size": args.train_size,
            "batch_size": args.batch_size,
            "train_portion": args.train_portion,
            "cutout": args.cutout,
            "cutout_length": args.cutout_length,
            "cutout_prob": args.cutout_prob,
        }

        with open(folder + f"/config_{args.predictor}_{i}.yaml", "w") as fh:
            yaml.dump(config, fh)


if __name__ == "__main__":
    """This is executed when run from the command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_seed", type=int, default=0, help="starting seed")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials")
    parser.add_argument("--predictor", type=str, default="snip", help="which predictor")
    parser.add_argument("--test_size", type=int, default=30, help="Test set size for predictor")
    parser.add_argument("--train_size", type=int, default=10, help="Train size if exp type is single")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Which dataset")
    parser.add_argument("--out_dir", type=str, default="run", help="Output directory")
    parser.add_argument("--search_space", type=str, default="nasbench201", help="nasbench201 or darts?")
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--train_portion", type=float, default=0.7, help="Size of training set")
    parser.add_argument("--cutout", action='store_true', default=False, help="Apply Cutout")
    parser.add_argument("--cutout_length", type=int, default=16, help="Cutout size")
    parser.add_argument("--cutout_prob", type=float, default=1.0, help="Cutout cut probability")
    args = parser.parse_args()
    main(args)
