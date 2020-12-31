import argparse
import os
import random
import yaml


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
        for i in range(args.start_seed, args.start_seed + args.trials):
            config = {
                'seed': i,
                'search_space': args.search_space,
                'dataset': args.dataset,
                'predictor': args.predictor,
                'test_size': args.test_size,
                'experiment_type': args.experiment_type,
                'train_size_list': args.train_size_list,
                'train_size_single': args.train_size_single,
                'fidelity_single': args.fidelity_single,
                'fidelity_list': args.fidelity_list,
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
                'search': {'predictor_type': args.predictor,
                           'epochs': args.epochs,
                           'checkpoint_freq': args.checkpoint_freq,
                           'fidelity': 200,
                           'sample_size': 10,
                           'population_size': 30,
                           'num_init': 10,
                           'k': 25,
                           'num_ensemble': 3,
                           'acq_fn_type': 'its',
                           'acq_fn_optimization': 'mutation',
                           'encoding_type': 'adjacency_one_hot',
                           'num_arches_to_mutate': 2,
                           'max_mutations': 1,
                           'num_candidates': 100,
                           'debug_predictor': False
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
    parser.add_argument("--dataset", type=str, default='cifar10', help="Which dataset")
    parser.add_argument("--out_dir", type=str, default='run', help="Output directory")
    parser.add_argument("--checkpoint_freq", type=int, default=5000, help="How often to checkpoint")
    parser.add_argument("--epochs", type=int, default=150, help="How many search epochs")
    parser.add_argument("--config_type", type=str, default='nas', help="nas or predictor?")
    parser.add_argument("--search_space", type=str, default='nasbench201', help="nasbench201 or darts?")

# nb201-reg
#    parser.add_argument("--train_size_list", type=list, default=[8, 12, 20, 32, 50, 80, 128, 203, 322, 512, 1000], help="train size list")
#    parser.add_argument("--train_size_single", type=int, default=100, help="Train size for single and vary_fidelity")
#    parser.add_argument("--fidelity_single", type=int, default=100, help="Fidelity for single and vary_train_size")
#    parser.add_argument("--fidelity_list", type=list, default=[5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, \
#                                                               110, 120, 130, 140, 150, 160, 170, 180, 190], help="train size list")

# nb201-lcsvr
#    parser.add_argument("--train_size_list", type=list, default=[8, 20, 50, 128, 322, 1000], help="train size list")
#    parser.add_argument("--train_size_single", type=int, default=100, help="Train size for single and vary_fidelity")
#    parser.add_argument("--fidelity_single", type=int, default=100, help="Fidelity for single and vary_train_size")
#    parser.add_argument("--fidelity_list", type=list, default=[5, 20, 40, 60, 80, 100, \
#                                                               120, 140, 160, 180, 190], help="train size list")
    
# darts
    parser.add_argument("--train_size_list", type=list, default=[8, 12, 20, 32, 50, 80, 128, 203, 322, 512], help="train size list")
    parser.add_argument("--train_size_single", type=int, default=512, help="Train size for single and vary_fidelity")
    parser.add_argument("--fidelity_single", type=int, default=50, help="Fidelity for single and vary_train_size")
    parser.add_argument("--fidelity_list", type=list, default=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, \
                                                               60, 65, 70, 75, 80, 85, 90, 95], help="train size list")
    parser.add_argument("--experiment_type", type=str, default='single', help="type of experiment")

    args = parser.parse_args()

    main(args)

