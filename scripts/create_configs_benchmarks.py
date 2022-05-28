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

    config = {
        'config_type': args.experiment,
        'seed': args.start_seed,
        'search_space': args.search_space,
        'dataset': args.dataset,
        'out_dir': args.out_dir,
        'predictor': args.predictor,
        'batch_size': args.batch_size,
        'train_portion': args.train_portion
    }

    with open(folder + f'/config_{args.start_seed}.yaml', 'w') as fh:
        yaml.dump(config, fh)


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_root", type=str, required=True, help="Root config directory")
    parser.add_argument("--start_seed", type=int, default=9000, help="Starting seed")
    parser.add_argument("--predictor", type=str, default='params', help="Predictor to evaluate")
    parser.add_argument("--dataset", type=str, default='cifar10', help="Which dataset")
    parser.add_argument("--out_dir", type=str, default='run', help="Output directory")
    parser.add_argument("--epochs", type=int, default=150, help="How many search epochs")
    parser.add_argument("--search_space", type=str, default='nasbench201', help="nasbench101/201/301/transnasbench101")
    parser.add_argument("--experiment", type=str, default='benchmarks', help="Experiment type")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--train_portion", type=float, default=0.7, help="Train portion")
    args = parser.parse_args()

    main(args)
