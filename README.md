<div align="center">
  <img src="images/naslib-logo.png" width="650" height="400">
</div>

<p align="center">
  <a href="https://github.com/automl/NASLib">
    <img src="https://img.shields.io/badge/Python-3.7%20%7C%203.8-blue?style=for-the-badge&logo=python" />
  </a>&nbsp;
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/pytorch-1.9-orange?style=for-the-badge&logo=pytorch" alt="PyTorch Version" />
  </a>&nbsp;
  <a href="https://github.com/automl/NASLib">
    <img src="https://img.shields.io/badge/open-source-9cf?style=for-the-badge&logo=Open-Source-Initiative" alt="Open Source" />
  </a>
  <a href="https://github.com/automl/NASLib">
    <img src="https://img.shields.io/github/stars/automl/naslib?style=for-the-badge&logo=github" alt="GitHub Repo Stars" />
  </a>
</p>


**NASLib** is a modular and flexible framework created with the aim of providing a common codebase to the community to facilitate research on **Neural Architecture Search** (NAS). It offers high-level abstractions for designing and reusing search spaces, interfaces to benchmarks and evaluation pipelines, enabling the implementation and extension of state-of-the-art NAS methods with a few lines of code. The modularized nature of NASLib
allows researchers to easily innovate on individual components (e.g., define a new
search space while reusing an optimizer and evaluation pipeline, or propose a new
optimizer with existing search spaces). It is designed to be modular, extensible and easy to use.

NASLib was developed by the [**AutoML Freiburg group**](https://www.automl.org/team/) and with the help of the NAS community, we are constantly adding new _search spaces_, _optimizers_ and _benchmarks_ to the library. Please reach out to zelaa@cs.uni-freiburg.de for any questions or potential collaborations. 

![naslib-overview](images/naslib-overall.png)

[**Setup**](#setup)
| [**Usage**](#usage)
| [**Docs**](examples/)
| [**Contributing**](#contributing)
| [**Cite**](#cite)

# Setup

While installing the repository, creating a new conda environment is recomended. [Install PyTorch GPU/CPU](https://pytorch.org/get-started/locally/) for your setup.

```bash
conda create -n mvenv python=3.7
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
```

Run setup.py file with the following command, which will install all the packages listed in [`requirements.txt`](requirements.txt)
```bash
pip install --upgrade pip setuptools wheel
pip install -e .
```

To validate the setup, you can run tests:

```bash
cd tests
coverage run -m unittest discover -v
```

The test coverage can be seen with `coverage report`.

# Usage

To get started, check out [`demo.py`](examples/demo.py).

```python
search_space = SimpleCellSearchSpace()

optimizer = DARTSOptimizer(config)
optimizer.adapt_search_space(search_space)

trainer = Trainer(optimizer, config)
trainer.search()        # Search for an architecture
trainer.evaluate()      # Evaluate the best architecture
```

For more examples see [naslib tutorial](examples/naslib_tutorial.ipynb), [intro to search spaces](examples/search_spaces.ipynb) and [intro to predictors](examples/predictors.md).

## Contributing
We welcome contributions to the library along with any potential issues or suggestions. Please create a pull request to the Develop branch.


## Cite

If you use this code in your own work, please use the following bibtex entries:

```bibtex
@misc{naslib-2020, 
  title={NASLib: A Modular and Flexible Neural Architecture Search Library}, 
  author={Ruchte, Michael and Zela, Arber and Siems, Julien and Grabocka, Josif and Hutter, Frank}, 
  year={2020}, publisher={GitHub}, 
  howpublished={\url{https://github.com/automl/NASLib}} }
 ``` 
 

<p align="center">
  <img src="images/predictors.png" alt="predictors" width="75%">
</p>

NASLib has been used to run an extensive comparison of 31 performance predictors (figure above). See the separate readme: <a href="docs/predictors.md">predictors.md</a>
and our paper: <a href="https://arxiv.org/abs/2104.01177">How Powerful are Performance Predictors in Neural Architecture Search?</a>

```bibtex
@article{white2021powerful,
  title={How Powerful are Performance Predictors in Neural Architecture Search?},
  author={White, Colin and Zela, Arber and Ru, Binxin and Liu, Yang and Hutter, Frank},
  journal={arXiv preprint arXiv:2104.01177},
  year={2021}
}
```
