import os
import json
import numpy as np

base_path = '/home/till/plotting_data/bbo/nasbench201/'
benchmark = 'ImageNet16-120/' #ImageNet16-120
base_path += benchmark

optimizers = {'dehb'}

optimizer_paths = []
for filename in os.listdir(base_path):
    optimizer_path = os.path.join(base_path, filename)
    if filename not in optimizers:
        continue
    if not os.path.isdir(optimizer_path):
        continue
    optimizer_paths.append(optimizer_path)

config_paths = []
for path in optimizer_paths:
    for filename in os.listdir(path):
        config_path = os.path.join(path, filename)
        if not os.path.isdir(config_path):
            continue
        config_paths.append(config_path)

result_files = []
for path in config_paths:
    for filename in os.listdir(path):
        result_file = os.path.join(path, filename, 'errors.json')
        if not os.path.isfile(result_file):
            continue
        result_files.append(result_file)

results = []
for result_file in result_files:
    f = open(result_file)
    results.append(json.load(f))

metric = 'valid_acc'
# metric = 'test_acc'

config_ids = 100
avgs = []
for id in range(config_ids):
    arch_seeds = [d for d in results if d[0]['config_id'] == id]
    avg_acc_seed = [max(d[1][metric]) for d in arch_seeds]
    avgs.append(np.mean(avg_acc_seed))
best_arch_avg_index = avgs.index(max(avgs))
best_arch_avg = max(avgs)


print("best archiecteure in average is config:{}, with: {} acc".format(best_arch_avg_index, best_arch_avg))
print("for the {} benchmark and the {} optimizer".format(benchmark, optimizers))

best_arch = max(results, key=lambda x:(x[1][metric]))
best_config_id = best_arch[0]['config_id']
best_arch_seeds = [d for d in results if d[0]['config_id'] == best_config_id]


worst_arch = min(results, key=lambda x:x[1][metric][-1])
default_archs = [d for d in results if d[0]['config_id'] == 0]
print(len(default_archs))
best_default_seed = max(default_archs, key=lambda x:x[1][metric][-1])
worst_default_seed = min(default_archs, key=lambda x:x[1][metric][-1])


print("-- Best Architecture --")
print("config_id: {}".format(best_arch[0]['config_id']))
print("Valid accuracy: {}".format(max(best_arch[1][metric])))
# [1][metric][-1]
print("Worst seed within architecture: {}".format(min(best_arch_seeds, key=lambda x:x[1]['valid_acc'][-1])[1][metric][-1]))
test = [d[1][metric][-1] for d in best_arch_seeds]
print("mean:{}".format(np.mean(test)))

print()
print("Default Architecture (without HPO)")
print("config_id: {}".format(best_default_seed[0]['config_id']))
print("Best Seed: Valid accuracy: {}".format(best_default_seed[1][metric][-1]))
print("Worst Seed: Valid accuracy: {}".format(worst_default_seed[1][metric][-1]))
test = [d[1][metric][-1] for d in default_archs]
print("mean:{}".format(np.mean(test)))

print()
print("-- Worst Architecture --")
print("config_id: {}".format(worst_arch[0]['config_id']))
print("Valid accuracy: {}".format(worst_arch[1][metric][-1]))


#results:
#sh for all try the config_0
#hb cifar 100: 85 cifar 10:  9   image_net: 65 
#bohb image net: 28   cifar 10: 59   cifar100: 60
#dehb: cifar100: 9    cifar10: 58      image_net: 87 
