# Getting Started with NASLib

In this guide, we demonstrate how to get started with the Neural Architecture Search Library (NASLib). NASLib provides a wide range of tools to facilitate neural architecture search and optimization. We will show you how to utilize various optimizers and search spaces, namely, DARTS optimizer with the NasBench301 search space, Regularized Evolution with NasBench201, and exploring zero-cost proxies with NasBench201. Let's dive into the specifics.

## Loading Configuration and Logging

```python
config = utils.get_config_from_args(config_type='nas')
logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)
utils.log_args(config)
```
The configuration parameters are loaded with ``utils.get_config_from_args(config_type='nas')``. If no other file is provided, it defaults to load from ``darts_defaults.yaml``. This configuration may include settings related to your dataset, architecture, optimization process, and so on. This flexibility allows easy switching between different optimizers and search spaces, as per the requirements of your experiments. The logger records this process for debugging or auditing.

## Defining a Subset of Available Optimizers and Search Spaces

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

## Preparing the Search Space and Optimizer

```python
dataset_api = get_dataset_api(config.search_space, config.dataset)
utils.set_seed(config.seed)
search_space = supported_search_spaces[config.search_space]
optimizer = supported_optimizers[config.optimizer]
optimizer.adapt_search_space(search_space, dataset=config.dataset, dataset_api=dataset_api)
```
This section sets the random seed for reproducibility, chooses the optimizer and search space as per the configuration, and adapts the optimizer to the chosen search space. It uses the dataset API to load the dataset corresponding to your selected search space.

## Running the Optimization Process

```python
trainer = Trainer(optimizer, config, lightweight_output=True)
trainer.search(resume_from="")
trainer.evaluate(resume_from="", dataset_api=dataset_api)
```
The Trainer object is created with the configured optimizer and starts the architecture search process. After searching, the trainer evaluates the best-found architecture. Both searching and evaluating can be resumed from previous checkpoints (specified as arguments to the respective functions). The 'lightweight_output' parameter, if set to True, will reduce the amount of output for each training epoch.

For the full code, see [getting_started](https://github.com/automl/NASLib/blob/Develop/examples/getting_started.py).




For more examples see [naslib tutorial](https://github.com/automl/NASLib/blob/Develop/examples/naslib_tutorial.ipynb), [intro to search spaces](https://github.com/automl/NASLib/blob/Develop/examples/search_spaces.ipynb) and [intro to predictors](https://github.com/automl/NASLib/blob/Develop/examples/predictors.md).
