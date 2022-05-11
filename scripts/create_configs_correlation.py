import argparse
import os
import yaml


def main(args):
    folder = os.path.join(
        args.config_root,
        args.experiment,
        args.predictor,
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
            'predictor': args.predictor,
            'test_size': args.test_size,
            'batch_size': args.batch_size,
            'cutout': args.cutout,
            'cutout_length': args.cutout_length,
            'cutout_prob': args.cutout_prob,
            'train_portion': args.train_portion
        }

        with open(folder + f'/config_{i}.yaml', 'w') as fh:
            yaml.dump(config, fh)


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_root", type=str, required=True, help="Root config directory")
    parser.add_argument("--start_seed", type=int, default=9000, help="Starting seed")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials")
    parser.add_argument("--predictor", type=str, default='params', help="Predictor to evaluate")
    parser.add_argument("--test_size", type=int, default=200, help="Test set size for predictor")
    parser.add_argument("--dataset", type=str, default='cifar10', help="Which dataset")
    parser.add_argument("--out_dir", type=str, default='run', help="Output directory")
    parser.add_argument("--epochs", type=int, default=150, help="How many search epochs")
    parser.add_argument("--search_space", type=str, default='nasbench201', help="nasbench101/201/301/transnasbench101")
    parser.add_argument("--experiment", type=str, default='correlation', help="Experiment type")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--cutout", type=bool, default=False, help="Cutout")
    parser.add_argument("--cutout_length", type=int, default=16, help="Cutout length")
    parser.add_argument("--cutout_prob", type=float, default=1.0, help="Cutout probability")
    parser.add_argument("--train_portion", type=float, default=0.7, help="Train portion")
    args = parser.parse_args()

    main(args)
