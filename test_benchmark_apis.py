import sys
import os
import argparse
import contextlib

from naslib.search_spaces import NasBench101SearchSpace, NasBench201SearchSpace, DartsSearchSpace, NasBenchNLPSearchSpace, NasBenchASRSearchSpace
from naslib.search_spaces.core.query_metrics import Metric
from naslib.utils import get_dataset_api


parser = argparse.ArgumentParser()
parser.add_argument('--search_space', required=False, type=str)
parser.add_argument('--dataset', required=False, type=str)
parser.add_argument('--all', required=False, action='store_true', help='Test all the benchmark APIs. Overrides --search_space and --dataset.')
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
    search_spaces = {
        'nasbench101': NasBench101SearchSpace,
        'nasbench201': NasBench201SearchSpace,
        'darts': DartsSearchSpace,
        'nlp': NasBenchNLPSearchSpace,
        'asr': NasBenchASRSearchSpace
    }

    datasets = {
        'nasbench101': ['cifar10'],
        'nasbench201': ['cifar10', 'cifar100', 'ImageNet16-120'],
        'darts': ['cifar10'],
        'nlp': ['treebank'],
        'asr': ['timit']
    }

    if args.all == True:
        success = []
        fail = []
        for space in search_spaces.keys():
            for dataset in datasets[space]:
                try:
                    print(f'Testing api for {space} search space with {dataset}...', end=" ", flush=True)
                    with nullify_all_output():
                        graph = search_spaces[space]()
                        result = test_api(graph, space, dataset, Metric.VAL_ACCURACY)
                    print('Success')
                except Exception as e:
                    print('Fail')
                    if args.show_error:
                        print(e)
    else:
        assert args.search_space is not None and args.dataset is not None, "Search space and dataset must be specified."
        try:
            print(f'Testing api for {args.search_space} with {args.dataset}...', end=" ", flush=True)
            with nullify_all_output():
                graph = search_spaces[args.search_space]()
                result = test_api(graph, args.search_space, args.dataset, Metric.VAL_ACCURACY)
            print('Success')
            print(result)
        except Exception as e:
            print('Fail')
            if args.show_error:
                print(e)


