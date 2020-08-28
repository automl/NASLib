# NASLib
NASLib is a Neural Architecture Search (NAS) library. Its purpose is to facilitate NAS research for the community by providing interfaces to several state-of-the-art NAS search spaces

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

If you plan to modify naslib consider adding the `-e` option when installing.

```
git clone ...
cd naslib
python install .
```

# Usage

```python
import naslib as nl

config = config_parser('naslib/configs/default.yaml')
parser = Parser('naslib/configs/default.yaml')

nl.utils.set_seed(config.seed)   # First thing to do to make naslib deterministic

search_space = nl.search_spaces.darts.DartsSearchSpace()

optimizer = nl.optimizers.oneshot.darts.DARTSOptimizer()
optimizer.adapt_search_space(search_space)  # The optimizer is prepating the search space

trainer = nl.optimizers.core.Trainer(optimizer, 'cifar10', config, parser)
trainer.train()
trainer.evaluate()
```




