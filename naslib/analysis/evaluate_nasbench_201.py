import glob
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

#rcParams.update({'figure.autolayout': True})
#plt.style.use('seaborn-whitegrid')
#mpl.use('Agg')
rcParams.update({'font.size': 12})

optimizer_dir_name_to_name = {
    'DARTS': 'DARTS', 'GDAS': 'GDAS', 'PCDARTS': 'PC-DARTS', 'SDARTSDARTS': 'Smooth-DARTS',
    'SDARTSPCDARTS': 'Smooth-PC-DARTS', 'SDARTSGDAS': 'Smooth-GDAS', 'gdas': 'GDAS', 'darts': 'DARTS', 're': 're'
}

markers={
        'SDARTSDARTS': '^',
        'SDARTSGDAS': 'v',
        'RS': 'D',
		'SDARTSPCDARTS': 'o',
		'gdas': 's',
		're': 'x',
        'True': '^',
		'Surrogate': 'h',
        'PC-DARTS': '^',
		'darts': 'h',
		'RandomNAS': 's',
        'HB': '>',
        'BOHB': '*',
        'TPE': '<'
}

def get_nb_eval(optimizer_runs, dataset, metric):
    nb_metric_per_run = []
    for run in optimizer_runs:
        nb_metric = []
        if dataset == 'darts':
            if metric == 'eval_acc1es':
                metric = 'test_acc'
            nb_metric = run[metric]
        else:
            for eval in run['test_acc']:
                nb_metric.append(eval[dataset][metric])
        nb_metric_per_run.append(nb_metric)
    return nb_metric_per_run


def analyze(optimizer_dict, dataset):
    fig, ax_left = plt.subplots(figsize=(6, 4))

    lines = []
    print(dataset)
    for optimizer_name, optimizer_runs in optimizer_dict.items():
        if not optimizer_runs:
            continue
        color = next(ax_left._get_lines.prop_cycler)['color']
        nb_test_error = 100 - np.array(get_nb_eval(optimizer_runs, dataset,
                                                 'eval_acc1es'))

        mean, std = np.mean(nb_test_error, axis=0), np.std(nb_test_error, axis=0)
        
        print("test! optimizer: {}, {:.02f} {{\\scriptsize $\\pm$ {:.02f}}}".format(optimizer_name, 100-mean[-1], std[-1]))
        line = ax_left.plot(np.arange(len(mean)), mean,
                     label="{}".format(optimizer_dir_name_to_name[optimizer_name]),
                     marker=markers.get(optimizer_name, None),
                     markersize=7, markevery=(0.1,0.1),
                     color=color)
        ax_left.fill_between(np.arange(len(mean)), mean - std, mean + std, alpha=0.3)
        lines.append(line[0])    
    
    if dataset=='cifar10':
        line = ax_left.plot([0, 50], [5.77, 5.77], label="RE final performance")
        lines.append(line[0])

    ax_left.set_xlabel('Search Epochs')
    ax_left.set_ylabel('Test Error [%]')
    if dataset == 'darts':
        plt.title("Results on Nas-Bench 301")
    else:
        plt.title(dataset)
    plt.grid(True, which="both", ls="-", alpha=0.8)
    ax_left.legend()

    if dataset == 'cifar10' or dataset == 'darts':
        ax_right = ax_left.twinx()  # instantiate a second axes that shares the same x-axis
        ax_right.set_ylabel('Validation Error [%]')
        
        for optimizer_name, optimizer_runs in optimizer_dict.items():
            if not optimizer_runs:
                continue
            color = next(ax_right._get_lines.prop_cycler)['color']
            one_shot_valid_error = 100 - np.array([run['valid_acc'] for run in
                                                optimizer_runs])
            mean, std = np.mean(one_shot_valid_error, axis=0), np.std(one_shot_valid_error, axis=0)
            print("validation! optimizer: {}, {:.02f} {{\\scriptsize $\\pm$ {:.02f}}}".format(optimizer_name, 100-mean[-1], std[-1]))
            line = ax_right.plot(np.arange(len(mean)), mean, 
                label="{} validation error".format(optimizer_dir_name_to_name[optimizer_name]),
                linestyle='-.', 
                alpha=0.3,
                color=color)
            ax_right.fill_between(np.arange(len(mean)), mean - std, mean + std, linestyle=':', alpha=0.1)
            lines.append(line[0])

    #ax_left.set_yscale('log')
    #ax_right.set_yscale('log')
    #plt.xlim(left=0, right=len(mean))
    plt.xlim(left=0, right=51 if dataset == 'darts' else 50)
    ax_left.legend(lines, [l.get_label() for l in lines])
    plt.tight_layout()
    plt.savefig('optim_{}_{}.pdf'.format(dir, dataset))

# dir = 'darts'
dir = 'nb201'

final = {
    'darts': [97.05, 96.99, 96.75, 97.3],
    'gdas': [96.93, 0, 0, 96.84],
}

# Darts 100 epoch val error
# 87.544
# 87.728
# 84.568
# 89.028 <- 4
# -----------
# 88.12  <- 5
# 87.328
# 85.356
# 87.628
# -----------
# 87.052
# 87.472
# 86.872
# 88.02  <- 12
# -----------
# 87.344
# 87.656
# 88.52  <- 15
# 86.84
# -----------

# gdas 100 epoch val error
# 87.76
# 88.08  <- 2
# 87.36
# 87.824
# -----------
# 87.824
# 88.164
# 88.484 <- 7
# 88.12
# -----------
# 88.244
# 88.596 <- 10
# 88.316
# 87.948
# -----------
# 88.248
# 87.656
# 88.364 <- 15
# 88.068
# -----------

if __name__ == '__main__':
    # optimizer_dict = {'DARTS': [], 'GDAS': [], 'PCDARTS': []}
    optimizer_dict = {'gdas': [], 'darts': [],} 
    for optimizer in optimizer_dict.keys():
        for res_json_path in glob.glob('{}/{}*.json'.format(dir, optimizer)):
            optimizer_dict[optimizer].append(json.load(open(res_json_path, 'r')))

    if dir == 'nb201':
        for dataset in ['cifar10', 'cifar100', 'ImageNet16-120']:
            analyze(optimizer_dict, dataset)
    else:
        analyze(optimizer_dict, 'darts')
