import argparse
import os
import yaml


def main(args):
    folder = os.path.join(
        args.config_root,
        args.experiment,
        f'train_size_{args.train_size}',
        f'k_{args.k}',
        f'{args.search_space}-{args.start_seed}',
        args.dataset
    )

    print(folder)
    os.makedirs(folder, exist_ok=True)
    args.start_seed = int(args.start_seed)
    args.trials = int(args.trials)

    for i in range(args.start_seed, args.start_seed + args.trials):
        config = {
            'config_type': args.experiment,
            'seed': i,
            'search_space': args.search_space,
            'dataset': args.dataset,
            'out_dir': args.out_dir,
            'predictor': args.k,
            'test_size': args.test_size,
            'train_size': args.train_size,
            'batch_size': args.batch_size,
            'cutout': args.cutout,
            'cutout_length': args.cutout_length,
            'cutout_prob': args.cutout_prob,
            'train_portion': args.train_portion,
            'zc_ensemble': args.zc_ensemble,
            'zc_names': args.zc_names,
            'zc_only': args.zc_only
        }

        with open(folder + f'/config_{i}.yaml', 'w') as fh:
            yaml.dump(config, fh)


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_root", type=str, required=True, help="Root config directory")
    parser.add_argument("--start_seed", type=int, default=9000, help="Starting seed")
    parser.add_argument("--trials", type=int, default=500, help="Number of trials")
    parser.add_argument("--predictor", type=str, default='xgb', help="Predictor to evaluate")
    parser.add_argument("--k", type=int, default=1, help="Number of proxies")
    parser.add_argument("--train_size", type=int, default=400, help="Train set size for predictor")
    parser.add_argument("--test_size", type=int, default=200, help="Test set size for predictor")
    parser.add_argument("--dataset", type=str, default='cifar10', help="Which dataset")
    parser.add_argument("--out_dir", type=str, default='run', help="Output directory")
    parser.add_argument("--epochs", type=int, default=150, help="How many search epochs")
    parser.add_argument("--search_space", type=str, default='nasbench201', help="nasbench101/201/301/transnasbench101")
    parser.add_argument("--experiment", type=str, default='xgb_correlation', help="Experiment type")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--cutout", type=bool, default=False, help="Cutout")
    parser.add_argument("--cutout_length", type=int, default=16, help="Cutout length")
    parser.add_argument("--cutout_prob", type=float, default=1.0, help="Cutout probability")
    parser.add_argument("--train_portion", type=float, default=0.7, help="Train portion")
    parser.add_argument("--zc_names", nargs='+', default=['params', 'flops', 'jacov', 'plain', 'grasp', 'snip', 'fisher', 'grad_norm', 'epe_nas', 'synflow', 'l2_norm'], help="Names of ZC predictors to use")
    
    parser.add_argument("--zc_ensemble", type=eval, choices=[True, False], default='True', help="True to use ensemble of ZC predictors")
    parser.add_argument("--zc_only", type=eval, choices=[True, False], default='True', help="Use only ZC features")

    args = parser.parse_args()

    print('args', args.zc_only, args.zc_ensemble)

    main(args)
