import sys

sys.path.append("../")

import os
import argparse
import contextlib

from naslib.search_spaces import (
        NasBench101SearchSpace,
        NasBench201SearchSpace,
        NasBench301SearchSpace,
        NasBenchNLPSearchSpace,
        NasBenchASRSearchSpace,
        TransBench101SearchSpaceMacro,
        TransBench101SearchSpaceMicro
    )
from naslib.search_spaces.core.query_metrics import Metric
from naslib.utils import get_dataset_api

search_spaces = {
    'nasbench101': NasBench101SearchSpace,
    'nasbench201': NasBench201SearchSpace,
    'nasbench301': NasBench301SearchSpace,
    'nlp': NasBenchNLPSearchSpace,
    'asr': NasBenchASRSearchSpace,
    'transbench101_micro': TransBench101SearchSpaceMicro,
    'transbench101_macro': TransBench101SearchSpaceMacro,
}

tasks = {
    'nasbench101': ['cifar10'],
    'nasbench201': ['cifar10', 'cifar100', 'ImageNet16-120', 'ninapro'],
    'nasbench301': ['cifar10'],
    'nlp': ['treebank'],
    'asr': ['timit'],
    'transbench101_micro': [
        'class_scene',
        'class_object',
        'jigsaw',
        'room_layout',
        'segmentsemantic',
        'normal',
        'autoencoder'
    ],
    'transbench101_macro': [
        'class_scene',
        'class_object',
        'jigsaw',
        'room_layout',
        'segmentsemantic',
        'normal',
        'autoencoder'
    ]
}


parser = argparse.ArgumentParser()
parser.add_argument('--search_space', required=False, type=str, help=f'API to test. Options: {list(search_spaces.keys())}')
parser.add_argument('--task', required=False, type=str)
parser.add_argument('--all', required=False, action='store_true', help='Test all the benchmark APIs. Overrides --search_space and --task.')
parser.add_argument('--show_error', required=False, action='store_true', help='Show the exception raised by the APIs if they crash.')
args = parser.parse_args()


@contextlib.contextmanager
def nullify_all_output():
    stdout = sys.stdout
    stderr = sys.stderr
    devnull = open(os.devnull, "w")
    try:
        sys.stdout = devnull
        sys.stderr = devnull
        yield
    finally:
        sys.stdout = stdout
        sys.stderr = stderr

def test_api(graph, search_space, dataset, metric):
    dataset_api = get_dataset_api(search_space=search_space, dataset=dataset)
    graph.sample_random_architecture(dataset_api)
    result = graph.query(metric, dataset=dataset, dataset_api=dataset_api)
    assert result != -1
    return result

if __name__ == '__main__':
    if args.all == True:
        success = []
        fail = []
        for space in search_spaces.keys():
            for task in tasks[space]:
                try:
                    print(f'Testing (search_space, task) api for ({space}, {task}) ...', end=" ", flush=True)
                    with nullify_all_output():
                        graph = search_spaces[space]()
                        result = test_api(graph, space, task, Metric.VAL_ACCURACY)
                    print('Success')
                except Exception as e:
                    print('Fail')
                    if args.show_error:
                        print(e)
    else:
        assert args.search_space is not None, "Search space must be specified."
        search_space_tasks = tasks[args.search_space] if args.task is None else [args.task]

        for task in search_space_tasks:
            try:
                print(f'Testing (search_space, task) api for ({args.search_space}, {task})...', end=" ", flush=True)
                # with nullify_all_output():
                graph = search_spaces[args.search_space]()
                result = test_api(graph, args.search_space, task, Metric.VAL_ACCURACY)
                print('Success')
            except Exception as e:
                print('Fail')
                if args.show_error:
                    print(e)


