# Getting Started with NASLib

In this guide, we demonstrate how to get started with the Neural Architecture Search Library (NASLib). NASLib provides a wide range of tools to facilitate neural architecture search and optimization. We will show you how to utilize various optimizers and search spaces, namely, DARTS optimizer with the NasBench301 search space, Regularized Evolution with NasBench201, and exploring zero-cost proxies with NasBench201. Let's dive into the specifics.

## DARTS and Regularized Evolution 
### Loading Configuration and Logging

```python
config = utils.get_config_from_args(config_type='nas')
logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)
utils.log_args(config)
```
The configuration parameters are loaded with ``utils.get_config_from_args(config_type='nas')``. If no other file is provided, it defaults to load from ``darts_defaults.yaml``. This configuration may include settings related to your dataset, architecture, optimization process, and so on. This flexibility allows easy switching between different optimizers and search spaces, as per the requirements of your experiments. The logger records this process for debugging or auditing.

### Defining a Subset of Available Optimizers and Search Spaces

```python
supported_optimizers = {
    're': RegularizedEvolution(config),
    'darts': DARTSOptimizer(**config),
}
supported_search_spaces = {
    'nasbench201': NasBench201SearchSpace(),
    'nasbench301': NasBench301SearchSpace(),
}
```
While NASLib supports a wide range of optimizers and search spaces, here we instantiate two specific optimizers (Regularized Evolution and DARTS) and two search spaces (NasBench201 and NasBench301). This demonstration shows how you can choose specific tools from the library based on your needs.

### Preparing the Search Space and Optimizer

```python
dataset_api = get_dataset_api(config.search_space, config.dataset)
utils.set_seed(config.seed)
search_space = supported_search_spaces[config.search_space]
optimizer = supported_optimizers[config.optimizer]
optimizer.adapt_search_space(search_space, dataset=config.dataset, dataset_api=dataset_api)
```
This section sets the random seed for reproducibility, chooses the optimizer and search space as per the configuration, and adapts the optimizer to the chosen search space. It uses the dataset API to load the dataset corresponding to your selected search space.

### Running the Optimization Process

```python
trainer = Trainer(optimizer, config, lightweight_output=True)
trainer.search(resume_from="")
trainer.evaluate(resume_from="", dataset_api=dataset_api)
```
The Trainer object is created with the configured optimizer and starts the architecture search process. After searching, the trainer evaluates the best-found architecture. Both searching and evaluating can be resumed from previous checkpoints (specified as arguments to the respective functions). The ``lightweight_output`` parameter, if set to True, will reduce the amount of output for each training epoch.

For the full code, see [getting_started](https://github.com/automl/NASLib/blob/Develop/examples/getting_started.py).


## Zero-cost 

NAS is typically compute-intensive because multiple models need to be evaluated before choosing the best one. To
reduce the computational power and time needed, a proxy task is often used for evaluating each model instead of full training. We can leverage Zero cost proxies as well as zero cost bechmarks to query the scores for NAS experiments. 

We have already downloaded the Zero Cost Benchmark API for NAS-Bench-201, which has the scores for all 15625 models evaluated for all three datasets (CIFAR-10, CIFAR-100 and ImageNet16-120) using all 13 Zero Cost proxies. 

### Setting up the experiment
```python
config = utils.get_config_from_args(config_type='zc')
logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)
utils.log_args(config)
utils.set_seed(config.seed)

supported_optimizers = {
    'bananas': Bananas,
    'npenas': Npenas,
}
supported_search_spaces = {
    'nasbench201': NasBench201SearchSpace,
    'nasbench301': NasBench301SearchSpace,
}
```
The experiment is setup the same way as before. But we will now choose ```config_type = 'zc'``` for loading config relevant to zero-cost experiment. Custom configs can be made the same way as the ```zc_config.yaml``` is written. We choose ```Bananas```  and ```Npenas``` as supported optimizers as these are only 2 optimizers which supports zero-cost API. It is also possible to integrate zero-cost API in custom optimizers' query function by replacing querying from NAS benchmarks with zero-cost API. 

### Preparing the loaders and zero-cost api 
```python
graph = supported_search_spaces[config.search_space]()
train_loader, val_loader, test_loader, train_transform, valid_transform = get_train_val_loaders(config)
zc_api = get_zc_benchmark_api(config.search_space, config.dataset)
```


### Querying using zero-cost predictor 
``` python
zc_pred = config.predictor 
graph.sample_random_architecture(dataset_api=dataset_api)
graph.parse()
zc_predictor = ZeroCost(method_type=zc_pred)
zc_score = zc_predictor.query(graph=graph, dataloader=train_loader)
```
You can query the zero-cost score by using a zero-cost predictor of your choice. 

### Correlation between zero-cost scores and validation accuracies
```python
val_accs = []
zc_scores = []
for _ in range(10):
    graph = supported_search_spaces[config.search_space]()
    graph.sample_random_architecture()
    graph.parse()
    acc = graph.query(metric=Metric.VAL_ACCURACY, dataset='cifar10',
                      dataset_api=dataset_api)
    val_accs.append(acc)
    zc_score = zc_predictor.query(graph=graph, dataloader=train_loader)
    zc_scores.append(zc_score)

corr_score = compute_scores(ytest=val_accs, test_pred=zc_scores)

print("The kendall-tau score is: ", corr_score["kendalltau"])
```

A handy feature to add in a custom optimizer is to compare the zero-cost scores against the validation accuracy by checking the correlation between them. We provide a handy utility function ```utils.compute_scores``` that gives 9 type of correlation scores. 

### Using zero-cost API
```python
zc_predictor = 'jacov'
spec = graph.get_hash()
zc_score = zc_api[str(spec)][zc_predictor]['score']
time_to_compute = zc_api[str(spec)][zc_predictor]['time'] 
```
We can save even further computation time by directly querying the zero-cost API instead of querying score by zero-cost predictors. 


### Training with zero-cost 
```python
graph = supported_search_spaces[config.search_space]()
optimizer = supported_optimizers[config.optimizer](config, zc_api=zc_api) 
optimizer.adapt_search_space(search_space, dataset_api=dataset_api)

trainer = Trainer(optimizer, config, lightweight_output=True)
trainer.search(resume_from="")
trainer.evaluate(resume_from="", dataset_api=dataset_api)
```

The 2 supported optimizers- ```Bananas``` and ```Npenas``` takes in ```zc_api``` parameter that uses zero-cost API while querying the scores for an architecture. Please look at ```zc_config.yaml``` for other relevant parameters required for using zero-cost scores.  


For more examples see [naslib tutorial](https://github.com/automl/NASLib/blob/Develop/examples/naslib_tutorial.ipynb), [intro to search spaces](https://github.com/automl/NASLib/blob/Develop/examples/search_spaces.ipynb) and [intro to predictors](https://github.com/automl/NASLib/blob/Develop/examples/predictors.md).
