import glob
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from util import get_trajectories, plot_losses
from IPython import embed

#rcParams.update({'figure.autolayout': True})
#plt.style.use('seaborn-whitegrid')
#mpl.use('Agg')
rcParams.update({'font.size': 12})

optimizer_dir_name_to_name = {
    'RE': 'Regularized Evolution',
    'RS': 'Random Search',
    'TPE': 'Tree Parzen Estimator'
}

lim = {
    'cifar10': 4e3,
    'cifar100': 4e3,
    'ImageNet16-120': 1e4
}

def get_nb_eval(optimizer_runs, dataset, metric):
    nb_metric_per_run = []
    for run in optimizer_runs:
        nb_metric = []
        last_val = 0 if metric == 'train_times' else -np.inf
        for eval in run['arch_eval']:
            curr_val = eval[dataset][metric]
            if metric == 'train_times':
                last_val += curr_val
                nb_metric.append(last_val)
            else:
                if curr_val > last_val:
                    nb_metric.append(curr_val)
                    last_val = curr_val
                else:
                    nb_metric.append(last_val)
        nb_metric_per_run.append(nb_metric)
    return nb_metric_per_run


def analyze(optimizer_dict, dataset):
    fig, ax = plt.subplots(1, figsize=(6, 4))
    parser_dict = {k: {} for k in optimizer_dict.keys()}

    for optimizer_name, optimizer_runs in optimizer_dict.items():
        nb_test_error = 100 - np.array(get_nb_eval(optimizer_runs, dataset,
                                                   'eval_acc1es'))
        parser_dict[optimizer_name]['losses'] = nb_test_error
        nb_training_time = np.array(get_nb_eval(optimizer_runs, dataset,
                                                'train_times'))
        parser_dict[optimizer_name]['time'] = nb_training_time

    trajectories = get_trajectories(parser_dict)
    plot_losses(fig, ax, None, trajectories, regret=False, plot_mean=True)

    ax.set_xlabel('Wallclock time [h]')
    ax.set_ylabel('Test Error [%]')
    #ax.set_yscale('log')
    ax.set_ylim(top=7)
    ax.set_xlim(left=1000)
    ax.set_xscale('log')
    plt.legend()
    plt.title(dataset)
    plt.grid(True, which="both", ls="-", alpha=0.8)
    plt.tight_layout()

    plt.savefig('optimizer_comp_nb201_{}_discrete.pdf'.format(dataset))
    #plt.show()


if __name__ == '__main__':
    optimizer_dict = {'RE': [], 'RS': []}
    # optimizer_dict = {'SDARTSDARTS': [], 'SDARTSGDAS': [], 'SDARTSPCDARTS': []}
    for optimizer in optimizer_dict.keys():
        optimizer_path = 'nb201/{}_*.json'.format(
            optimizer)
        for res_json_path in glob.glob(optimizer_path):
            optimizer_dict[optimizer].append(json.load(open(res_json_path, 'r')))

    for dataset in ['cifar10', 'cifar100', 'ImageNet16-120']:
        analyze(optimizer_dict, dataset)


# final test results.
# RE: 94.15, 94.36, 94.04, 94.36
# RS: 94.15, 94.29, 94.36, 93.87

re_test = [94.15, 94.36, 94.04, 94.36]
re_vali = [91.36, 91.57, 91.38, 91.57]
rs_test = [94.15, 94.29, 94.36, 93.87]
rs_vali = [91.36, 91.21, 91.57, 91.02]

#gdas = [96.93, 96.8, 97.0, 96.84]

#hierarchical
gdas = [95.49, 95.45, 95.35]
darts = [95.7, 95.57, 95.75]

def test(x):
    mean = np.mean(x)
    std = np.std(x)
    print("{:.02f} {{\\scriptsize $\\pm$ {:.02f}}}".format(mean, std))

test(darts)


#test(re_test)
#test(rs_test)
#print()
#test(re_vali)
#test(rs_vali)