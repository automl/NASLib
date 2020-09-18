# NASLib
NASLib is a Neural Architecture Search (NAS) library. Its purpose is to facilitate NAS research for the community by providing interfaces to several state-of-the-art NAS search spaces

> :warning: **This library is under construction** and there is no official release yet. Feel 
> free to play around and have a look but be aware that the *APIs will be changed* until we have a first release.

It is designed to be modular, extensible and easy to use.

# Usage

```python
search_space = SimpleCellSearchSpace()

optimizer = DARTSOptimizer(config)
optimizer.adapt_search_space(search_space)

trainer = Trainer(optimizer, 'cifar10', config)
trainer.search()        # Search for an architecture
trainer.evaluate()      # Evaluate the best architecture
```

For an example file see `examples`.

# Requirements

Make sure you use the latest version of pip. It makes sense to set up a virtual environment, too.

```
python3 -m venv naslib
source naslib/bin/activate

pip install --upgrade pip setuptools wheel
pip install cython
```

# Installation

Clone and install.

If you plan to modify naslib consider adding the `-e` option for `pip install`.

```
git clone ...
cd naslib
pip install .
```




