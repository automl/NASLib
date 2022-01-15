import os
import json
# from matplotlib.lines import _LineStyle
import matplotlib.pyplot as plt

from collections import defaultdict

def get_results(predictor, path, filename, metric='valid_acc'):
    """
    Get statistics for successive halving
    TODO: make metric selectable, currently 'val_acc' is fixed 
    """
    algo_path = os.path.join(path, predictor)
    for seed_dir in os.listdir(algo_path):
        result_file = os.path.join(algo_path, seed_dir, filename)
        result = json.load(open(result_file))
        return result

def plot_sh():
    """
    Plots learning curves for successive halving
    """
    folder = os.path.expanduser('./run/cifar10/nas_predictors/nasbench201')
    predictor = 'var_sparse_gp'
    results = get_results(predictor, folder, 'successivehalving_stats.json', metric='test_acc')

    for arch, stats in results.items():
        x = stats['fidelity']
        values = stats['val_acc']
        plt.plot(x, values, linestyle='-', label=arch)
    plt.xscale('log', base=2)
    plt.rcParams['grid.linestyle'] = 'dotted'
    plt.show()
    plt.savefig('plot_sh_nb201.pdf', bbox_inches = 'tight', pad_inches = 0.1)

def plot_hb():
    """
    Plots learning curves for hyperband
    """
    folder = os.path.expanduser('./run/cifar10/nas_predictors/nasbench201')
    predictor = 'var_sparse_gp'
    results = get_results(predictor, folder, 'hyperband_stats.json', metric='test_acc')
    s = len(results)
    
    figure, axis = plt.subplots(s, 1)
    for sh, sh_stats in results.items():
        for arch, stats in sh_stats.items():
            x = stats['fidelity']
            values = stats['val_acc']
            axis[int(sh)].plot(x, values, linestyle='-', label=arch, marker='x')
            axis[int(sh)].set_xscale('log', base=3)
    plt.show()
    plt.savefig('plot_hb_nb201.pdf', bbox_inches = 'tight', pad_inches = 0.1)


if __name__ == '__main__':
    plot_sh()