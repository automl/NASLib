import glob
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})
plt.style.use('seaborn-whitegrid')
mpl.use('Agg')

optimizer_dir_name_to_name = {
    'RE': 'Regularized Evolution',
    'RS': 'Random Search',
    'TPE': 'Tree Parzen Estimator'
}


def get_nb_eval(optimizer_runs, dataset, metric):
    nb_metric_per_run = []
    for run in optimizer_runs:
        nb_metric = []
        last_val = -np.inf
        for eval in run['arch_eval']:
            curr_val = eval[dataset][metric]
            if curr_val > last_val:
                nb_metric.append(curr_val)
                last_val = curr_val
            else:
                nb_metric.append(last_val)
        nb_metric_per_run.append(nb_metric)
    return nb_metric_per_run


def analyze(optimizer_dict, dataset):
    fig = plt.figure(figsize=(5, 4))
    plt.ylabel('Test Error (NB) (-)')

    for optimizer_name, optimizer_runs in optimizer_dict.items():
        nb_test_error = 1 - np.array(get_nb_eval(optimizer_runs, dataset, 'test_accuracy')) / 100

        mean, std = np.mean(nb_test_error, axis=0), np.std(nb_test_error, axis=0)
        plt.plot(np.arange(len(mean)), mean, label=optimizer_dir_name_to_name[optimizer_name])
        plt.fill_between(np.arange(len(mean)), mean - std, mean + std, alpha=0.3)
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)

    plt.yscale('log')
    plt.xlim(left=0, right=200)
    plt.ylim(top=6e-1)
    plt.savefig('optimizer_comp_nb201_{}_discrete.pdf'.format(dataset))


if __name__ == '__main__':
    optimizer_dict = {'RE': [], 'RS': [], 'TPE': []}
    # optimizer_dict = {'SDARTSDARTS': [], 'SDARTSGDAS': [], 'SDARTSPCDARTS': []}
    for optimizer in optimizer_dict.keys():
        optimizer_path = '/home/siemsj/projects/NASLib/naslib/benchmarks/nasbench201/run/cifar10/{}'.format(
            optimizer)
        for res_json_path in glob.glob(os.path.join(optimizer_path, 'errors_*.json')):
            optimizer_dict[optimizer].append(json.load(open(res_json_path, 'r')))

    for dataset in ['cifar10', 'cifar100', 'ImageNet16-120']:
        analyze(optimizer_dict, dataset)
