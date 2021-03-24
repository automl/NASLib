import argparse
import os
import random
import yaml

import numpy as np


def main(args):

    if args.config_type == 'nas':
        folder = f'{args.out_dir}/{args.dataset}/configs/nas'
        os.makedirs(folder, exist_ok=True)
        args.start_seed = int(args.start_seed)
        args.trials = int(args.trials)

        for i in range(args.start_seed, args.start_seed + args.trials):
            config = {
                'seed': i,
                'search_space': args.search_space,
                'dataset': args.dataset,
                'optimizer': args.optimizer,
                'out_dir': args.out_dir,
                'search': {'checkpoint_freq': args.checkpoint_freq,
                           'epochs': args.epochs,
                           'fidelity': 200,
                           'sample_size': 10,
                           'population_size': 30,
                           'num_init': 10,
                           'k': 25,
                           'num_ensemble': 3,
                           'acq_fn_type': 'its',
                           'acq_fn_optimization': 'mutation',
                           'encoding_type': 'path',
                           'num_arches_to_mutate': 2,
                           'max_mutations': 1,
                           'num_candidates': 100,
                           'predictor_type':'feedforward',
                           'debug_predictor': False
                          }
            }

            if args.optimizer == 'lcsvr' and args.experiment_type == 'vary_fidelity':
                path = folder + f'/config_{args.optimizer}_train_{i}.yaml'
            if args.optimizer == 'lcsvr' and args.experiment_type == 'vary_train_size':
                path = folder + f'/config_{args.optimizer}_fidelity_{i}.yaml'
            else:
                path = folder + f'/config_{args.optimizer}_{i}.yaml'

            with open(path, 'w') as fh:
                yaml.dump(config, fh)

    elif args.config_type == 'predictor':
        folder = f'{args.out_dir}/{args.dataset}/configs/predictors'
        os.makedirs(folder, exist_ok=True)
        args.start_seed = int(args.start_seed)
        args.trials = int(args.trials)
        
        if args.search_space == 'nasbench101':
            total_epochs = 108 - 1
            max_train_size = 1000
        elif args.search_space == 'nasbench201':
            total_epochs = 200 - 1
            max_train_size = 1000
        elif args.search_space == 'darts':
            total_epochs = 96 - 1
            max_train_size = 500
        elif args.search_space == 'nlp':
            total_epochs = 50 - 1
            max_train_size = 1000

        train_size_list = [int(j) for j in np.logspace(start=np.log(5.1)/np.log(2), 
                                                       stop=np.log(max_train_size)/np.log(2), 
                                                       num=11, endpoint=True, base=2.0)]
        fidelity_list = [int(j) for j in np.logspace(start=0.9, 
                                                     stop=np.log(total_epochs)/np.log(2), 
                                                     num=15, endpoint=True, base=2.0)]

        if args.search_space == 'nlp':
            fidelity_list.pop(2)
            fidelity_list.insert(5, 6)

        if 'svr' in args.predictor:
            train_size_list.pop(0)
            fidelity_list.pop(0)
            fidelity_list.pop(0)

        elif 'omni' in args.predictor and args.search_space != 'darts':
            train_size_list.pop(0)
            train_size_list.pop(-1)
            fidelity_list.pop(1)

        elif 'omni' in args.predictor and args.search_space == 'darts':
            train_size_list.pop(0)
            train_size_list.pop(-1)
            fidelity_list.pop(1)
            fidelity_list.pop(1)

        for i in range(args.start_seed, args.start_seed + args.trials):
            config = {
                'seed': i,
                'search_space': args.search_space,
                'dataset': args.dataset,
                'out_dir': args.out_dir,
                'predictor': args.predictor,
                'test_size': args.test_size,
                'uniform_random': args.uniform_random,
                'experiment_type': args.experiment_type,
                'train_size_list': train_size_list,
                'train_size_single': args.train_size_single,
                'fidelity_single': args.fidelity_single,
                'fidelity_list': fidelity_list,
                'max_hpo_time': 900,
            }

            with open(folder + f'/config_{args.predictor}_{i}.yaml', 'w') as fh:
                yaml.dump(config, fh)

    elif args.config_type == 'nas_predictor':
        folder = f'{args.out_dir}/{args.dataset}/configs/nas_predictors'
        os.makedirs(folder, exist_ok=True)
        args.start_seed = int(args.start_seed)
        args.trials = int(args.trials)

        for i in range(args.start_seed, args.start_seed + args.trials):
            config = {
                'seed': i,
                'search_space': args.search_space,
                'dataset': args.dataset,
                'optimizer': args.optimizer,
                'out_dir': args.out_dir,
                'search': {'predictor_type': args.predictor,
                           'epochs': args.epochs,
                           'checkpoint_freq': args.checkpoint_freq,
                           'fidelity': 200,
                           'sample_size': 10,
                           'population_size': 30,
                           'num_init': 20,
                           'k': 20,
                           'num_ensemble': 3,
                           'acq_fn_type': 'its',
                           'acq_fn_optimization': 'random_sampling',
                           'encoding_type': 'adjacency_one_hot',
                           'num_arches_to_mutate': 5,
                           'max_mutations': 1,
                           'num_candidates': 200,
                           'batch_size': 256,
                           'data_size': 25000,
                           'cutout': False,
                           'cutout_length': 16,
                           'cutout_prob': 1.0,
                           'train_portion': 0.7
                          }
            }

            path = folder + f'/config_{args.optimizer}_{args.predictor}_{i}.yaml'
            with open(path, 'w') as fh:
                yaml.dump(config, fh)

    else:
        print('invalid config type in create_configs.py')


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    parser.add_argument("--start_seed", type=int, default=0, help="starting seed")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials")
    parser.add_argument("--optimizer", type=str, default='rs', help="which optimizer")
    parser.add_argument("--predictor", type=str, default='full', help="which predictor")
    parser.add_argument("--test_size", type=int, default=30, help="Test set size for predictor")
    parser.add_argument("--uniform_random", type=int, default=1, help="Train/test set generation type (bool)")
    parser.add_argument("--train_size_single", type=int, default=10, help="Train size if exp type is single")
    parser.add_argument("--fidelity_single", type=int, default=5, help="Fidelity if exp type is single")
    parser.add_argument("--dataset", type=str, default='cifar10', help="Which dataset")
    parser.add_argument("--out_dir", type=str, default='run', help="Output directory")
    parser.add_argument("--checkpoint_freq", type=int, default=5000, help="How often to checkpoint")
    parser.add_argument("--epochs", type=int, default=150, help="How many search epochs")
    parser.add_argument("--config_type", type=str, default='nas', help="nas or predictor?")
    parser.add_argument("--search_space", type=str, default='nasbench201', help="nasbench201 or darts?")
    parser.add_argument("--experiment_type", type=str, default='single', help="type of experiment")

    args = parser.parse_args()

    main(args)
