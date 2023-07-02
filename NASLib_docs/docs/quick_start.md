# Usage

To get started, check out [`demo.py`](https://github.com/automl/NASLib/blob/Develop/examples/demo.py).

```python
search_space = SimpleCellSearchSpace()

optimizer = DARTSOptimizer(**config.search)
optimizer.adapt_search_space(search_space, config.dataset)

trainer = Trainer(optimizer, config)
trainer.search()        # Search for an architecture
trainer.evaluate()      # Evaluate the best architecture
```

For more examples see [naslib tutorial](https://github.com/automl/NASLib/blob/Develop/examples/naslib_tutorial.ipynb), [intro to search spaces](https://github.com/automl/NASLib/blob/Develop/examples/search_spaces.ipynb) and [intro to predictors](https://github.com/automl/NASLib/blob/Develop/examples/predictors.md).
