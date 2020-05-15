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
    'DARTS': 'DARTS', 'GDAS': 'GDAS', 'PCDARTS': 'PC-DARTS', 'SDARTSDARTS': 'Smooth-DARTS',
    'SDARTSPCDARTS': 'Smooth-PC-DARTS', 'SDARTSGDAS': 'Smooth-GDAS'
}


def get_nb_eval(optimizer_runs, dataset, metric):
    nb_metric_per_run = []
    for run in optimizer_runs:
        nb_metric = []
        for eval in run['arch_eval']:
            nb_metric.append(eval[dataset][metric])
        nb_metric_per_run.append(nb_metric)
    return nb_metric_per_run


def analyze(optimizer_dict, dataset):
    fig, ax_left = plt.subplots(figsize=(5, 4))
    ax_left.set_ylabel('Test Error (NB) (-)')

    for optimizer_name, optimizer_runs in optimizer_dict.items():
        nb_test_error = 1 - np.array(get_nb_eval(optimizer_runs, dataset, 'test_accuracy')) / 100

        mean, std = np.mean(nb_test_error, axis=0), np.std(nb_test_error, axis=0)
        ax_left.plot(np.arange(len(mean)), mean, label=optimizer_dir_name_to_name[optimizer_name])
        ax_left.fill_between(np.arange(len(mean)), mean - std, mean + std, alpha=0.3)
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)

    ax_right = ax_left.twinx()  # instantiate a second axes that shares the same x-axis
    ax_right.set_ylabel('Validation Error (OS) (-.-)')
    for optimizer_name, optimizer_runs in optimizer_dict.items():
        one_shot_valid_error = 1 - np.array([run['valid_acc'] for run in optimizer_runs]) / 100
        mean, std = np.mean(one_shot_valid_error, axis=0), np.std(one_shot_valid_error, axis=0)
        ax_right.plot(np.arange(len(mean)), mean, linestyle='-.', alpha=0.4)
        ax_right.fill_between(np.arange(len(mean)), mean - std, mean + std, linestyle=':', alpha=0.1)

    ax_left.set_yscale('log')
    ax_right.set_yscale('log')
    plt.xlim(left=0, right=len(mean))
    plt.savefig('optimizer_comp_nb201_{}.pdf'.format(dataset))


if __name__ == '__main__':
    # optimizer_dict = {'DARTS': [], 'GDAS': [], 'PCDARTS': []}
    optimizer_dict = {'SDARTSDARTS': [], 'SDARTSGDAS': [], 'SDARTSPCDARTS': []}
    for optimizer in optimizer_dict.keys():
        optimizer_path = '/home/siemsj/projects/NASLib/naslib/benchmarks/nasbench201/run/cifar10/{}Optimizer'.format(
            optimizer)
        for res_json_path in glob.glob(os.path.join(optimizer_path, 'errors_*.json')):
            optimizer_dict[optimizer].append(json.load(open(res_json_path, 'r')))

    for dataset in ['cifar10', 'cifar100', 'ImageNet16-120']:
        analyze(optimizer_dict, dataset)
