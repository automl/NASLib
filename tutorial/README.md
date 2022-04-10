# Tutorial

This tutorial assumes that you are completely new to NASLib, and will introduce you to the core ideas and APIs of the library. By the end of it, you will know everything you need to know to make a submission to the [Zero-Cost NAS Competition](link here).

## Introduction to Search Spaces

Let's begin by taking a look [NAS-Bench-201](https://arxiv.org/abs/2001.00326) search space as an example:

<img src="./images/nb201_arch.png" alt="alt text" title="image Title" width="700"/>

As you can see, the architecture consists of multiple *cells* stacked together with residual blocks in between which downsample the feature maps. Each cell has 6 edges, each of which could hold one of 5 operations from a predefined operation set.

Let's first create a search space in NASLib:


```python
from naslib.search_spaces import NasBench201SearchSpace
search_space = NasBench201SearchSpace(n_classes=10)
search_space
```




    Graph makrograph-0.8281313, scope None, 20 nodes



Equivalently, you could also create the search space as follows:


```python
from naslib.search_spaces import get_search_space
search_space = get_search_space(name='nasbench201', dataset='cifar10')
search_space
```




    Graph makrograph-0.3794643, scope None, 20 nodes



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
graph = search_space.clone() # Clone the search space first
cell = graph.edges[2, 3]['op']
show_only_one_edge = True

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
    Operation on edge (1, 2): AvgPool1x1(
      (avgpool): AvgPool2d(kernel_size=3, stride=1, padding=1)
    )
    Operation on edge (1, 3): Identity()
    Operation on edge (1, 4): Identity()
    Operation on edge (2, 3): AvgPool1x1(
      (avgpool): AvgPool2d(kernel_size=3, stride=1, padding=1)
    )
    Operation on edge (2, 4): Identity()
    Operation on edge (3, 4): ReLUConvBN(
      (op): Sequential(
        (0): ReLU()
        (1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    
    Architecture encoding: (4, 0, 0, 4, 0, 2)


To create the PyTorch model, one needs only to parse this `Graph` as follows:


```python
import torch

# Create the graph
graph = search_space.clone()
graph.sample_random_architecture()

# Parse the graph
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
    ), Graph cell-0.5183977, scope stage_1, 4 nodes, Graph cell-0.9655811, scope stage_1, 4 nodes, Graph cell-0.8979745, scope stage_1, 4 nodes, Graph cell-0.7528428, scope stage_1, 4 nodes, Graph cell-0.0031009, scope stage_1, 4 nodes, ResNetBasicblock(
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
    ), Graph cell-0.9865748, scope stage_2, 4 nodes, Graph cell-0.0930538, scope stage_2, 4 nodes, Graph cell-0.2040329, scope stage_2, 4 nodes, Graph cell-0.5242195, scope stage_2, 4 nodes, Graph cell-0.6348156, scope stage_2, 4 nodes, ResNetBasicblock(
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
    ), Graph cell-0.1557997, scope stage_3, 4 nodes, Graph cell-0.8948010, scope stage_3, 4 nodes, Graph cell-0.3674761, scope stage_3, 4 nodes, Graph cell-0.7841433, scope stage_3, 4 nodes, Graph cell-0.5250999, scope stage_3, 4 nodes, Sequential(
      (op): Sequential(
        (0): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): ReLU(inplace=True)
        (2): AdaptiveAvgPool2d(output_size=1)
        (3): Flatten(start_dim=1, end_dim=-1)
        (4): Linear(in_features=64, out_features=10, bias=True)
      )
    )]
    
    Result: tensor([[-0.4965,  0.4482,  0.4002,  0.1254, -0.1047, -0.0755,  0.0543, -0.0041,
             -0.4122,  0.4150]], grad_fn=<AddmmBackward>)


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

    Compact model representation is: (4, 3, 4, 1, 0, 3)
    NAS-Bench-201 representation is: |avg_pool_3x3~0|+|nor_conv_1x1~0|none~1|+|avg_pool_3x3~0|skip_connect~1|nor_conv_1x1~2|
    Validation accuracy: 84.36
    


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


The Kendall Tau correlation of the predicted and actual scores is the metric of interest in the competition.


```python
from scipy import stats
stats.kendalltau(scores, actual_scores)
```

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

For the submission to be complete, the class `ZeroCostPredictor` must be saved in a file named `predictor.py` and zipped together with an empty `metdata` file.


```python
# Directory structure:
# sample_submission/
# |__ metadata        # empty file
# |__ submission.py   # File with zero-cost predictor code
!touch sample_submission/metadata
!zip sample_submission.zip sample_submission/*
```

You can use this `sample_submission.zip` as a test submission on CodaLab.


```python

```
