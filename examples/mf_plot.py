import os
import json
import matplotlib.pyplot as plt

from collections import defaultdict

def get_results(predictor, path, epochs, metric='valid_acc', dataset='cifar10', ug=False):
    """
    Get statistics for successive halving
    # TODO: make metric selectable, currently 'val_acc' is fixed 
    """
    algo_path = os.path.join(path, predictor)
    for seed_dir in os.listdir(algo_path):
        result_file = os.path.join(algo_path, seed_dir, 'sh_stats.json')
        result = json.load(open(result_file))
        return result

def plot_sh():
    """Plots successive halving learning curves
    """
    # set up parameters for the experiments
    epochs = 300

    folder = os.path.expanduser('./run/cifar10/nas_predictors/nasbench201')
    predictor = 'var_sparse_gp'
    results = get_results(predictor, folder, epochs=epochs, metric='test_acc', ug=True)

    for arch, results in results.items():
        x = results['fidelity']
        values = results['val_acc']
        plt.plot(x, values, linestyle='-', label=arch)
    plt.rcParams['grid.linestyle'] = 'dotted'
    plt.show()
    plt.savefig('plot_nb201.pdf', bbox_inches = 'tight', pad_inches = 0.1)

if __name__ == '__main__':
    plot_sh()