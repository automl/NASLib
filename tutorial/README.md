# Tutorial

This tutorial assumes that you are completely new to NASLib, and will introduce you to the core ideas and APIs of the library. By the end of it, you will know everything you need to know to make a submission to the [Zero-Cost NAS Competition](https://codalab.lisn.upsaclay.fr/competitions/3932).

## Introduction to Search Spaces

Let's begin by taking a look at the [NAS-Bench-201](https://arxiv.org/abs/2001.00326) search space as an example:

<img src="./images/nb201_arch.png" alt="alt text" title="image Title" width="700"/>

As you can see, the architecture consists of multiple *cells* stacked together with residual blocks in between which downsample the feature maps. Each cell has 6 edges, each of which could hold one of 5 operations from a predefined operation set.

Let's first create a search space in NASLib:


```python
from naslib.search_spaces import NasBench201SearchSpace
search_space = NasBench201SearchSpace(n_classes=10)
search_space
```




    Graph makrograph-0.1983816, scope None, 20 nodes



Equivalently, you could also create the search space as follows:


```python
from naslib.search_spaces import get_search_space
search_space = get_search_space(name='nasbench201', dataset='cifar10')
search_space
```




    Graph makrograph-0.6036780, scope None, 20 nodes



`Graphs` in NASLib inherit from both [PyTorch Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) as well as [NetworkX DiGraph](https://networkx.org/documentation/stable/reference/classes/digraph.html). It uses [NetworkX](https://networkx.org/) to create the architecture structure, which can later be parsed to create a standard PyTorch Module.

Here's a closer look at the search_space graph that we just created:


```python
print('Nodes in the graph:', search_space.nodes())
print('Edges in the graph:', search_space.edges())
```

    Nodes in the graph: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    Edges in the graph: [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20)]


Every node and edge of the `Graph` can hold operations, which could be either a concrete implementation of the NASLib `AbstractPrimitive`, or another `Graph`. 


```python
print('Operation on edge 1-2 of the graph:', search_space.edges[1, 2]['op']) # Concrete implementation of AbstractPrimitive as 'op' on the edge
print('Operation on edge 2-3 of the graph:', search_space.edges[2, 3]['op']) # NASLib Graph as 'op' on the edge
```

    Operation on edge 1-2 of the graph: Stem(
      (seq): Sequential(
        (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    Operation on edge 2-3 of the graph: Graph named 'cell' with 4 nodes and 6 edges


Here's a closer look at a cell, which is itself another `Graph`:


```python
cell = search_space.edges[2, 3]['op']
print('Nodes in the cell:', cell.nodes())
print('Edges in the cell:', cell.edges())
```

    Nodes in the cell: [1, 2, 3, 4]
    Edges in the cell: [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]


The cell graph thus has the same structure as the cell shown in the NAS-Bench-201 architecture figure. Let's now look at what is present on the edges of the cell.


```python
print(f'Operations on edge 1-2 of cell:')
print(cell.edges[1, 2]['op'])
```

    Operations on edge 1-2 of cell:
    [Identity(), Zero (stride=1), ReLUConvBN(
      (op): Sequential(
        (0): ReLU()
        (1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    ), ReLUConvBN(
      (op): Sequential(
        (0): ReLU()
        (1): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    ), AvgPool1x1(
      (avgpool): AvgPool2d(kernel_size=3, stride=1, padding=1)
    )]


All edges on the cell have a list of candidate operations on them, as seen above. The candidate operations here are Identity, Zero, ReLU-3x3Convolution-BatchNorm, ReLU-1x1Convolution-BatchNorm, and a 1x1AveragePool. You can uncomment and run the next lines to see all the edges.


```python
# for edge in cell.edges:
#     print('%'*50)
#     print(f'Operations on edge 1-2 of cell:')
#     print(cell.edges[edge]['op'])
```

## Sampling random models

Let's now sample a random architecture from the search space.


```python
# A model can be sampled from a search space Graph only once
# So clone the search space first
graph = search_space.clone()
cell = graph.edges[2, 3]['op']

# Initially, the operation on an edge is simply a list of all candidate operations
print('Before sampling operation')
print('Operation on edge 1-2:', cell.edges[1, 2]['op'])

# Sample a random architecture
graph.sample_random_architecture()

# After sampling, it is replaced by one operation from the list
# This means you cannot invoke sample_random_architecture()
# on the same graph twice
print('\nAfter sampling operation')
for edge in cell.edges():
    print(f'Operation on edge {edge}:', cell.edges[edge]['op'])

# Get the representation of this architecture:
print('\nArchitecture encoding:', graph.get_hash())
```

    Before sampling operation
    Operation on edge 1-2: [Identity(), Zero (stride=1), ReLUConvBN(
      (op): Sequential(
        (0): ReLU()
        (1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    ), ReLUConvBN(
      (op): Sequential(
        (0): ReLU()
        (1): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    ), AvgPool1x1(
      (avgpool): AvgPool2d(kernel_size=3, stride=1, padding=1)
    )]
    
    After sampling operation
    Operation on edge (1, 2): Identity()
    Operation on edge (1, 3): ReLUConvBN(
      (op): Sequential(
        (0): ReLU()
        (1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    Operation on edge (1, 4): Zero (stride=1)
    Operation on edge (2, 3): Zero (stride=1)
    Operation on edge (2, 4): ReLUConvBN(
      (op): Sequential(
        (0): ReLU()
        (1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    Operation on edge (3, 4): Identity()
    
    Architecture encoding: (0, 2, 1, 1, 2, 0)


To create the PyTorch model, one needs only to parse this `Graph` as follows:


```python
import torch

# Create the graph
graph = search_space.clone()
graph.sample_random_architecture()

# Parse the NASLib graph and make it a PyTorch model
print('Sub modules in the graph before parsing', list(graph.children()))
graph.parse()
print('\nSub modules in the graph after parsing', list(graph.children()))

# Test the graph with a forward pass of a small random minibatch
result = graph(torch.randn(1, 3, 32, 32))
print('\nResult:', result)
```

    Sub modules in the graph before parsing []
    
    Sub modules in the graph after parsing [Stem(
      (seq): Sequential(
        (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    ), Graph cell-0.5767110, scope stage_1, 4 nodes, Graph cell-0.4516561, scope stage_1, 4 nodes, Graph cell-0.7715191, scope stage_1, 4 nodes, Graph cell-0.8664710, scope stage_1, 4 nodes, Graph cell-0.1674664, scope stage_1, 4 nodes, ResNetBasicblock(
      (conv_a): ReLUConvBN(
        (op): Sequential(
          (0): ReLU()
          (1): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (conv_b): ReLUConvBN(
        (op): Sequential(
          (0): ReLU()
          (1): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (downsample): Sequential(
        (0): AvgPool2d(kernel_size=2, stride=2, padding=0)
        (1): Conv2d(16, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
      )
    ), Graph cell-0.2991547, scope stage_2, 4 nodes, Graph cell-0.9848268, scope stage_2, 4 nodes, Graph cell-0.4333180, scope stage_2, 4 nodes, Graph cell-0.8190599, scope stage_2, 4 nodes, Graph cell-0.0986330, scope stage_2, 4 nodes, ResNetBasicblock(
      (conv_a): ReLUConvBN(
        (op): Sequential(
          (0): ReLU()
          (1): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (conv_b): ReLUConvBN(
        (op): Sequential(
          (0): ReLU()
          (1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (downsample): Sequential(
        (0): AvgPool2d(kernel_size=2, stride=2, padding=0)
        (1): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      )
    ), Graph cell-0.7796141, scope stage_3, 4 nodes, Graph cell-0.6157184, scope stage_3, 4 nodes, Graph cell-0.0597902, scope stage_3, 4 nodes, Graph cell-0.6364680, scope stage_3, 4 nodes, Graph cell-0.9865471, scope stage_3, 4 nodes, Sequential(
      (op): Sequential(
        (0): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): ReLU(inplace=True)
        (2): AdaptiveAvgPool2d(output_size=1)
        (3): Flatten(start_dim=1, end_dim=-1)
        (4): Linear(in_features=64, out_features=10, bias=True)
      )
    )]
    
    Result: tensor([[ 0.2043,  0.1416, -0.4398, -0.2264, -0.2273,  0.3394,  0.2632, -0.2477,
              0.2011, -0.0592]], grad_fn=<AddmmBackward>)


## Querying the performance of a model

NAS-Bench-201 has 15,625 models in its search space, all of which have been evaluated for classification task on three separate datasets - CIFAR10, CIFAR100, and ImageNet-16-120. Given the architecture encoding of a model, one can simply query the benchmark to see its final performance for any one of these tasks.

The first thing to do is to ensure that you have the benchmark data files. For now, we will download and test only the NAS-Bench-201 benchmark for the CIFAR10 task. 

If you've already downloaded the data and tested the API, you can skip this step. If these scripts do not work for you, please follow the instructions [here](https://github.com/automl/NASLib/tree/automl-conf-competition/dataset_preparation#-dataset-preparation) to get the data.


```python
!chmod +x ../scripts/bash_scripts/download_data.sh

!echo Downloading data for NASBench201 API
!cd .. && source scripts/bash_scripts/download_data.sh nb201 cifar10

!echo Testing NASBench201 API
!cd .. && python test_benchmark_apis.py --search_space nasbench201 --task cifar10 --show_error
```

    Downloading data for NASBench201 API
    dataset = cifar10
    search_space = nb201
    cifar10 exists
    Testing NASBench201 API
    Testing (search_space, task) api for (nasbench201, cifar10)... Success



```python
# Convenience function to sample a new model from the given search space and parse it
def sample_and_parse_graph(name='nasbench201', dataset='cifar10'):
    search_space = get_search_space(name, dataset)
    search_space.sample_random_architecture()
    search_space.parse()
    return search_space
```

Now, lets sample a model from the search space and query its validation performance.


```python
# First, load the benchmark API.
from naslib.utils import get_dataset_api
dataset_api = get_dataset_api('nasbench201', 'cifar10')

# Sample a random architecture model
graph = sample_and_parse_graph()

# Show architecture encoding
from naslib.search_spaces.nasbench201.conversions import convert_naslib_to_str

print(f'Compact model representation is: {graph.get_hash()}')
print(f'NAS-Bench-201 representation is: {convert_naslib_to_str(graph)}')

# Query the benchmark
from naslib.search_spaces.core.query_metrics import Metric
val_accuracy = graph.query(
    metric=Metric.VAL_ACCURACY,
    dataset='cifar10',
    dataset_api=dataset_api
)

print(f'Validation accuracy: {val_accuracy}\n')
```

    Compact model representation is: (2, 0, 0, 3, 3, 2)
    NAS-Bench-201 representation is: |nor_conv_3x3~0|+|skip_connect~0|nor_conv_1x1~1|+|skip_connect~0|nor_conv_1x1~1|nor_conv_3x3~2|
    Validation accuracy: 90.62
    


## Zero Cost Predictors
Let's now move on to trying the zero cost predictors already available in NASLib.


```python
""" Evaluates a ZeroCost predictor for a search space and dataset/task"""
from naslib.predictors import ZeroCost
from naslib.utils import utils
import numpy as np

# Get the configs from naslib/configs/predictor_config.yaml (and the command line arguments, if any)
# The configs include the zero-cost method to use, the search space and dataset/task to use, amongst others.
# For now, we will manually update the config here
config = utils.get_config_from_args()
# print(config)
config.search_space='nasbench201'
config.dataset='cifar10'

# Initialize the predictor
# Method type can be "fisher", "grasp", "grad_norm", "jacov", "snip", "synflow", "flops" or "params"
predictor = ZeroCost(method_type=config.predictor)

# Create the models to score
n = 10
print(f'Sampling {n} models')
models = [sample_and_parse_graph() for i in range(n)]

# Get the dataloader for this dataset
train_loader, val_loader, test_loader, train_transform, val_transform = utils.get_train_val_loaders(config=config)

# Score each model
print('Scoring models with predictor')
scores = [predictor.query(model, dataloader=test_loader) for model in models]

# Query benchmarks to get the actual scores
print('Querying benchmarks for actual scores')
actual_scores = [
        model.query(
            metric=Metric.VAL_ACCURACY,
            dataset='cifar10',
            dataset_api=dataset_api
        ) for model in models
    ]

print('Done.')
```

    Sampling 10 models
    Files already downloaded and verified
    Files already downloaded and verified
    Scoring models with predictor
    Querying benchmarks for actual scores
    Done.


The Kendall Tau correlation of the predicted and actual scores is the metric of interest in the competition.


```python
from scipy import stats
stats.kendalltau(scores, actual_scores)
```




    KendalltauResult(correlation=0.5555555555555555, pvalue=0.02860945767195767)



To make the evaluation of predictors more convenient, `ZeroCostPredictorEvaluator` class is provided to you.


```python
from naslib.evaluators.zc_evaluator import ZeroCostPredictorEvaluator
from naslib.utils import setup_logger
import logging

# Set up logger
logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)

# Change the default test_size in configs
config.test_size = 14

# Initialize the ZeroCostPredictorEvaluator class
predictor_evaluator = ZeroCostPredictorEvaluator(predictor, config=config)
predictor_evaluator.adapt_search_space(search_space, dataset_api=dataset_api)

# Evaluate the predictor
results = predictor_evaluator.evaluate()
results[-1]['kendalltau']
```

    [32m[04/11 09:48:49 nl.evaluators.zc_evaluator]: [0mSampling from search space...
    [32m[04/11 09:48:52 nl.evaluators.zc_evaluator]: [0mQuerying the predictor
    Files already downloaded and verified
    Files already downloaded and verified
    [32m[04/11 09:49:02 nl.evaluators.zc_evaluator]: [0mCompute evaluation metrics
    [32m[04/11 09:49:02 nl.evaluators.zc_evaluator]: [0mdataset: cifar10, predictor: synflow, kendalltau 0.5604
    [32m[04/11 09:49:02 nl.evaluators.zc_evaluator]: [0mmae: 2.7995851399501432e+40, rmse: 1.0437419559137192e+41, pearson: 0.2368, spearman: 0.7451, kendalltau: 0.5604, kt_2dec: 0.5604, kt_1dec: 0.5604, precision_10: 0.9, precision_20: 0.6, full_ytest: [86.77 71.92 85.18 89.47 87.4  84.85 85.19 89.01 85.17 89.89 87.12 84.12
     89.44 85.69], full_testpred: [2.10175186e+19 2.66518826e+06 2.94391780e+25 2.12203568e+35
     1.63192861e+31 2.52076873e+32 2.08993446e+23 1.95421532e+26
     5.09783416e+15 1.41177900e+39 1.96238489e+19 2.56321128e+06
     3.90529928e+41 1.55693468e+18], query_time: 0.7541, 





    0.5604395604395604



## Sample Submission

Now, you're ready to create a sample submission for the competition. In this example, we're going to write a very simple zero-cost predictor, which simply counts the number of parameters in the model. The predictor thus assigns higher score to bigger models.


```python
from naslib.search_spaces.core.graph import Graph
from naslib.predictors.predictor import Predictor
from torch.utils.data import DataLoader

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

class ZeroCostPredictor(Predictor):
    """ A sample submission class.

    CodaLab will import this class from your submission.py, and evaluate it.
    Your class must be named `ZeroCostPredictor`, and must contain self.method_type
    with your method name in it.
    """
    def __init__(self):
        self.method_type = 'MyZeroCostPredictorName'

    def pre_process(self) -> None:
        """ This method is called exactly once before query is called repeatedly with the models to score """
        pass

    def query(self, graph: Graph, dataloader:DataLoader=None) -> float:
        """ Predict the score of the given model

        Args:
            graph       : Model to score
            dataloader  : DataLoader for the task to predict. E.g., if the task is to
                          predict the score of a model for classification on CIFAR10 dataset,
                          a CIFAR10 Dataloader is passed here.

        Returns:
            Score of the model. Higher the score, higher the model is ranked.
        """

        # You can consume the dataloader and pass the data through the model here.
        # Uncomment the following lines to try this

        # data, labels = next(iter(dataloader))
        # logits = graph(data)

        # In this example, however, we simply count the number of parameters in the model
        # This zero-cost predictor thus gives higher score to larger models.
        score = count_parameters(graph)

        # Higher the score, higher the ranking of the model
        return score

```

You can evaluate the performance of this predictor across all the benchmarks (search-space/dataset combination) available in NASLib as follows. 

*Make sure you set up the data for NAS-Bench-201 with CIFAR100 and ImageNet16-120 before you run the next cell. Manual download instructions [here](https://github.com/automl/NASLib/tree/automl-conf-competition/dataset_preparation#-dataset-preparation)*.


```python
# This function evaluates a predictor with test_size number of models across the benchmarks available for 
# the given search space
from naslib.evaluators import full_evaluate_predictor

predictor = ZeroCostPredictor()

# You can also test the zero cost predictors already available in NASLib
# from naslib.predictors import ZeroCost
# predictor = ZeroCost(method_type='synflow')

all_benchmarks = {
    'nasbench201': ['cifar10', 'cifar100', 'ImageNet16-120'],
    'nasbench301': ['cifar10'],
    'transbench101_micro': ['class_scene', 'class_object', 'jigsaw'],
}


# search_spaces = ["nasbench201", "nasbench301", "transbench101_micro"]
full_evaluate_predictor(predictor, test_size=10, search_spaces=["nasbench201"])
```

    [32m[04/11 09:49:02 nl.utils.utils]: [0mCommand line args: Namespace(config_file=None, opts=[], datapath=None)
    [32m[04/11 09:49:02 nl.utils.utils]: [0mExperiment dir : run/cifar10/predictors/synflow/1000
    [32m[04/11 09:49:02 nl.utils.utils]: [0mExperiment dir : run/cifar10/predictors/synflow/1000/search
    [32m[04/11 09:49:02 nl.utils.utils]: [0mExperiment dir : run/cifar10/predictors/synflow/1000/eval
    [32m[04/11 09:49:03 nl.evaluators.zc_evaluator]: [0mSampling from search space...
    [32m[04/11 09:49:05 nl.evaluators.zc_evaluator]: [0mQuerying the predictor
    Files already downloaded and verified
    Files already downloaded and verified
    [32m[04/11 09:49:09 nl.evaluators.zc_evaluator]: [0mCompute evaluation metrics
    [32m[04/11 09:49:09 nl.evaluators.zc_evaluator]: [0mdataset: cifar10, predictor: MyZeroCostPredictorName, kendalltau 0.4243
    [32m[04/11 09:49:09 nl.evaluators.zc_evaluator]: [0mmae: 425572.515, rmse: 518474.3217, pearson: 0.5307, spearman: 0.5124, kendalltau: 0.4243, kt_2dec: 0.4243, kt_1dec: 0.4243, precision_10: 0.9, precision_20: 0.45, full_ytest: [86.77 71.92 85.18 89.47 87.4  84.85 85.19 89.01 85.17 89.89], full_testpred: [ 129306  101306  559386  559386  587386  587386  129306  400346  129306
     1073466], query_time: 0.3104, 
    [32m[04/11 09:49:10 nl.evaluators.zc_evaluator]: [0mSampling from search space...
    [32m[04/11 09:49:12 nl.evaluators.zc_evaluator]: [0mQuerying the predictor
    Files already downloaded and verified
    Files already downloaded and verified
    [32m[04/11 09:49:16 nl.evaluators.zc_evaluator]: [0mCompute evaluation metrics
    [32m[04/11 09:49:16 nl.evaluators.zc_evaluator]: [0mdataset: cifar100, predictor: MyZeroCostPredictorName, kendalltau 0.4243
    [32m[04/11 09:49:16 nl.evaluators.zc_evaluator]: [0mmae: 431444.4, rmse: 523304.2823, pearson: 0.5585, spearman: 0.4939, kendalltau: 0.4243, kt_2dec: 0.4243, kt_1dec: 0.4243, precision_10: 0.9, precision_20: 0.45, full_ytest: [65.32 46.62 62.5  69.1  64.4  63.04 62.52 68.52 63.36 70.62], full_testpred: [ 135156  107156  565236  565236  593236  593236  135156  406196  135156
     1079316], query_time: 0.3593, 
    [32m[04/11 09:49:17 nl.evaluators.zc_evaluator]: [0mSampling from search space...
    [32m[04/11 09:49:19 nl.evaluators.zc_evaluator]: [0mQuerying the predictor
    [32m[04/11 09:49:26 nl.evaluators.zc_evaluator]: [0mCompute evaluation metrics
    [32m[04/11 09:49:26 nl.evaluators.zc_evaluator]: [0mdataset: ImageNet16-120, predictor: MyZeroCostPredictorName, kendalltau 0.5185
    [32m[04/11 09:49:26 nl.evaluators.zc_evaluator]: [0mmae: 432772.44, rmse: 524398.9466, pearson: 0.6353, spearman: 0.673, kendalltau: 0.5185, kt_2dec: 0.5185, kt_1dec: 0.5185, precision_10: 0.9, precision_20: 0.45, full_ytest: [35.9333 14.9333 35.4    40.7333 39.4333 34.8333 33.9    42.1667 33.5667
     44.7   ], full_testpred: [ 136456  108456  566536  566536  594536  594536  136456  407496  136456
     1080616], query_time: 0.7046, 
    nasbench201              ||cifar10                  ||0.4242640687119285
    nasbench201              ||cifar100                 ||0.4242640687119285
    nasbench201              ||ImageNet16-120           ||0.5185449728701348
    Average Kendall-Tau: 0.4556910367646639


For the submission to be complete, the class `ZeroCostPredictor` must be saved in a file named `submission.py` and zipped together with an empty `metdata` file.


```python
# Directory structure:
# sample_submission/
# |__ metadata        # empty file
# |__ submission.py   # File with zero-cost predictor code
!touch sample_submission/metadata
!zip sample_submission.zip sample_submission/*
```

      adding: sample_submission/metadata (stored 0%)
      adding: sample_submission/submission.py (deflated 57%)


You can use this `sample_submission.zip` as a test submission on CodaLab.


```python

```
