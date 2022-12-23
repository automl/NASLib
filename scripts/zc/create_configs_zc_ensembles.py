import argparse
import os
import yaml

def main(args):
    folder = os.path.join(
        args.config_root,
        args.config_type,
        "zc_only" if args.zc_only else "zc_architecture",
        "use_zc_api" if args.use_zc_api else "use_predictor",
        args.optimizer,
        f'{args.search_space}-{args.start_seed}',
        args.dataset
    )
    print(folder)
    os.makedirs(folder, exist_ok=True)
    args.start_seed = int(args.start_seed)
    args.trials = int(args.trials)

    for i in range(args.start_seed, args.start_seed + args.trials):
        config = {
            'seed': i,
            'search_space': args.search_space,
            'dataset': args.dataset,
            'optimizer': args.optimizer,
            'config_type': args.config_type,
            'predictor': args.predictor,
            'out_dir': args.out_dir,
            'test_size': args.test_size,
            'train_portion': args.train_portion,
            'batch_size': args.batch_size,
            'cutout': args.cutout,
            'cutout_length': args.cutout_length,
            'cutout_prob': args.cutout_prob,
        }

        config_keys = set(config.keys())
        args_keys = set([arg for arg in vars(args)])
        search_args = args_keys.difference(config_keys)

        search_config = {arg:getattr(args, arg) for arg in search_args}
        del(search_config['config_root'])
        del(search_config['trials'])
        del(search_config['start_seed'])
        search_config['seed'] = i

        config['search'] = search_config
        with open(folder + f'/config_{i}.yaml', 'w') as fh:
            yaml.dump(config, fh)

if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    parser.add_argument("--search_space", type=str, default='nasbench201', help="nasbench101/201/301/tnb101")
    parser.add_argument("--dataset", type=str, default='cifar10', help="Which dataset")
    parser.add_argument("--optimizer", type=str, default='npenas', help="Blackbox optimizer to use")
    parser.add_argument("--predictor", type=str, default='zc', help="which predictor")

    parser.add_argument("--out_dir", type=str, default='run', help="Output directory")
    parser.add_argument("--start_seed", type=int, default=9000, help="starting seed")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials")

    parser.add_argument("--test_size", type=int, default=1000, help="Test set size for predictor")
    parser.add_argument("--train_portion", type=float, default=0.7, help="Train portion")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--cutout", type=bool, default=False, help="Cutout")
    parser.add_argument("--cutout_length", type=int, default=16, help="Cutout length")
    parser.add_argument("--cutout_prob", type=float, default=1.0, help="Cutout probability")
    parser.add_argument("--config_root", type=str, default='configs', help="Root config directory")

    # Search options
    parser.add_argument("--epochs", type=int, default=200, help="Number of search epochs")
    parser.add_argument("--fidelity", type=int, default=-1, help="Number of epochs")
    parser.add_argument("--sample_size", type=int, default=10, help="Number of samples")
    parser.add_argument("--population_size", type=int, default=50, help="Number of individuals")
    parser.add_argument("--checkpoint_freq", type=int, default=1001, help="Checkpoint frequency")
    parser.add_argument("--zc_names", nargs='+', default=['params', 'flops', 'jacov', 'plain', 'grasp', 'snip', 'fisher', 'grad_norm', 'epe_nas', 'synflow', 'l2_norm'], help="Names of ZC predictors to use")
    parser.add_argument("--k", type=int, default=10, help="Top k candidates to choose in each batch")
    parser.add_argument("--num_init", type=int, default=10, help="Root config directory")
    parser.add_argument("--num_ensemble", type=int, default=1, help="Root config directory")
    parser.add_argument("--acq_fn_type", type=str, default='its', help="Root config directory")
    parser.add_argument("--acq_fn_optimization", type=str, default='mutation', help="Root config directory")
    parser.add_argument("--encoding_type", type=str, default='adjacency', help="Root config directory")
    parser.add_argument("--num_arches_to_mutate", type=int, default=2, help="Root config directory")
    parser.add_argument("--max_mutations", type=int, default=1, help="Root config directory")
    parser.add_argument("--num_candidates", type=int, default=100, help="Root config directory")
    parser.add_argument("--predictor_type", type=str, default='xgb', help="Root config directory")

    parser.add_argument("--config_type", type=str, default='zc_and_adjacency', help="Type of experiment")
    parser.add_argument("--zc_ensemble", type=bool, default=True, help="True to use ensemble of ZC predictors")
    parser.add_argument("--zc", type= lambda x : (True if x == "True" else False), default=True, help="Whether zerocost is used or not")
    parser.add_argument("--zc_only", type= lambda x : (True if x == "True" else False), default=False, help="Specify how to define zerocost features into the tree")
    parser.add_argument("--use_zc_api", type= lambda x : (True if x == "True" else False), default=False, help="Whether to use zc_api (True) or zc_predictor (False)")

    args = parser.parse_args()

    main(args)
