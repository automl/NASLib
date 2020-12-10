
import argparse
import os
import random
from pathlib import Path
import yaml


def main(args):

    if args.config_type == 'nas':
        folder = f'{args.out_dir}/{args.dataset}/configs/nas'
        Path(folder).mkdir(exist_ok=True)
        args.start_seed = int(args.start_seed)
        args.trials = int(args.trials)

        for i in range(args.start_seed, args.start_seed + args.trials):
            config = {
                'seed': i,
                'dataset': args.dataset,
                'optimizer': args.optimizer,
                'search': {'checkpoint_freq': args.checkpoint_freq,
                           'epochs': args.epochs,
                           'fidelity': 200,
                           'sample_size': 10,
                           'population_size': 30,
                           'num_init': 10,
                           'k': 10,
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

            with open(folder + f'/config_{args.optimizer}_{i}.yaml', 'w') as fh:
                yaml.dump(config, fh)
                
    elif args.config_type == 'predictor':
        folder = f'{args.out_dir}/{args.dataset}/configs/predictors'
        Path(folder).mkdir(exist_ok=True)
        args.start_seed = int(args.start_seed)
        args.trials = int(args.trials)

        for i in range(args.start_seed, args.start_seed + args.trials):
            config = {
                'seed': i,
                'dataset': args.dataset,
                'predictor': args.predictor,
                'train_size': args.train_size,
                'test_size': args.test_size
            }

            with open(folder + f'/config_{args.predictor}_{i}.yaml', 'w') as fh:
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
    parser.add_argument("--train_size", type=int, default=30, help="Training set size for predictor")    
    parser.add_argument("--test_size", type=int, default=30, help="Test set size for predictor")    
    parser.add_argument("--dataset", type=str, default='cifar10', help="Which dataset")
    parser.add_argument("--out_dir", type=str, default='run', help="Output directory")    
    parser.add_argument("--checkpoint_freq", type=int, default=5000, help="How often to checkpoint")
    parser.add_argument("--epochs", type=int, default=150, help="How many search epochs")
    parser.add_argument("--config_type", type=str, default='nas', help="nas or predictor?")
    
    
    args = parser.parse_args()

    main(args)
    
    
