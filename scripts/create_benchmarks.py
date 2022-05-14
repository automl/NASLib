import os
import fnmatch
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--search_space', required=False, default='nasbench201', type=str)
parser.add_argument('--config-file', required=False, default='asdasd', type=str)
args = parser.parse_args()

def print_neat(items):
    for i in items:
        print(i)

def find_files(src, fname):
    matches = []
    for root, dirnames, filenames in os.walk(src):
        for filename in fnmatch.filter(filenames, fname):
            matches.append(os.path.join(root, filename))

    return matches

if __name__ == '__main__':
    N_MODELS_PER_SEED = 100
    benchmark_search_space = args.search_space
    print(benchmark_search_space)

    files = find_files(f'run/xgb_correlation/{benchmark_search_space}', 'benchmark.json')

    filtered_files = []
    search_spaces = []
    datasets = []

    for f in files:
        components = f.split('/')
        search_space, dataset, n_models, seed = components[-5], components[-4], int(components[-3]), components[-2]

        if n_models == N_MODELS_PER_SEED and search_space == benchmark_search_space:
            search_spaces.append(search_space)
            datasets.append(dataset)
            filtered_files.append(f)

    print_neat(filtered_files)

    data = {}
    for search_space, dataset, file in zip(search_spaces, datasets, filtered_files):
        if search_space not in data:
            data[search_space] = {}
            data[search_space][dataset] = {}
        elif dataset not in data[search_space]:
            data[search_space][dataset] = {}

        with open(file) as f:
            zc_benchmarks = json.load(f)[0]

        data[search_space][dataset].update(zc_benchmarks)

    results_file = f'naslib/data/zc_{search_space}.json'
    with open(results_file, 'w') as f:
        json.dump(data, f)

    print()
    for dataset in set(datasets):
        print(f'Number of archs for {dataset}: {len(data[search_space][dataset])}')

    print(f'Saved benchmark file to {results_file}')
    print('\nDone.')