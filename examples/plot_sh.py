#%%
import sys
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import pickle
import pandas as pd

from collections import defaultdict

# needed for plotting
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import FormatStrFormatter

def restructure_results(errors :dict):
    """This super fancy function restructures the results in order plot successive halving correctly
    NOTE: This in not optimimzed at all!
    NOTE: this function should be obsolete thus we save the stats now in the correct format.
    """
    print(errors[1].keys())
    len_errors = len(errors[1]['train_acc'])
    train_acc = errors[1]['train_acc']
    valid_acc = errors[1]['valid_acc']
    fidelity = errors[1]['fidelity']
    archs = errors[1]['arch']
    test_acc = errors[1]['test_acc']
    runtime = errors[1]['runtime']
    train_time = errors[1]['train_time']
    new_errors = defaultdict(lambda: defaultdict(list))

    for idx in range(len_errors):
        new_errors[archs[idx]]['train_acc'].append(train_acc[idx])
        new_errors[archs[idx]]['valid_acc'].append(valid_acc[idx])
        new_errors[archs[idx]]['fidelity'].append(fidelity[idx])
        new_errors[archs[idx]]['test_acc'].append(test_acc[idx])
        new_errors[archs[idx]]['runtime'].append(runtime[idx])
        new_errors[archs[idx]]['train_time'].append(train_time[idx])
    return new_errors

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

def plot_sh(new_results :dict):
    """Plots successive halving learning curves
    for key, data_list in data_dict.items():
        dates_str, values = zip(*data_list)  # Unpack
        dates = convert_str_to_dates(dates_str)  # Not handled in example
        plt.plot(dates, values, label=key)
    plt.legend()
    """
    for arch, results in new_results.items():
        x = results['fidelity']
        values = results['val_acc']
        plt.plot(x, values, linestyle='-', label=arch)
    plt.rcParams['grid.linestyle'] = 'dotted'
    plt.show()
    plt.savefig('plot_nb201.pdf', bbox_inches = 'tight', pad_inches = 0.1)


def main():
    """
    Just plots the successive halving
    """

    # set up parameters for the experiments
    epochs = 300

    folder = os.path.expanduser('./run/cifar10/nas_predictors/nasbench201')
    predictor = 'var_sparse_gp'
    results = get_results(predictor, folder, epochs=epochs, metric='test_acc', ug=True)


    plot_sh(results)

if __name__ == '__main__':
    main()
# %%
