# How Powerful are Performance Predictors in Neural Architecture Search?

[How Powerful are Performance Predictors in Neural Architecture Search?](https://arxiv.org/abs/2104.01177)\
Colin White, Arber Zela, Robin Ru, Yang Liu, and Frank Hutter.\
_arXiv:2104.01177_.

Dozens of techniques have been proposed to predict the final performance of neural architectures, however, it is not well-understood how different families of techniques compare to one another. We give the first large-scale study of performance predictors by analyzing 31 techniques ranging from learning curve extrapolation, to weight-sharing, to supervised learning, to "zero-cost" proxies. We test correlation- and rank-based performance measures in a variety of settings, as well as the ability of each technique to speed up predictor-based NAS frameworks. We show that certain families of predictors can be combined to achieve even better predictive power.
<p align="center">
  <img src="../images/predictors.png" alt="predictors" width="90%">
</p>

# Installation

Follow the installation instructions from the <a href="../README.md">main NASLib readme</a>.

## Install nasbench301 from source
```bash
git clone git@github.com:automl/nasbench301.git
cd nasbench301
cat requirements.txt | xargs -n 1 -L 1 pip install
pip install .
```

## Download all datasets
```bash
cd naslib/data
```
First download nasbench301 v1.0 from [here](https://figshare.com/articles/software/nasbench301_models_v1_0_zip/13061510), unzip it, and rename the top-level folder to nb301_models. So the full path to the models is `NASLib/naslib/data/nb301_models/xgb_v1.0/...`

Download nasbench101, nasbench201 (all three datasets), and nasbench301 training data from [here](https://drive.google.com/drive/folders/1rwmkqyij3I24zn5GSO6fGv2mzdEfPIEa?usp=sharing).

# Usage

There are to types of experiments: stand-alone predictor experiments, and predictor-based NAS experiments. Thanks to the NAS-Bench datasets, almost all of our experiments can be run on CPUs (indeed, we used CPUs for nearly all of our experiments). The two weight-sharing predictors (OneShot and RS-WS) were the only predictors where we used GPUs, since they require retraining NAS-Bench models from scratch.

## Stand-alone predictor experiments
To run a single test on a predictor, modify the settings in `naslib/benchmarks/predictors/predictor_config.yaml` as you desire (e.g., change the search space, predictor, train_size (initialization time), and fidelity (query time). Then run
```bash
python naslib/benchmarks/predictors/runner.py --config-file naslib/benchmarks/predictors/predictor_config.yaml
```
To run 100 trials of all performance predictors on some search space (e.g. darts), run
```bash
python naslib/benchmarks/predictors/run_darts.sh
```

## Predictor-based NAS experiments
To run a single test on a NAS algorithm, modify the settings in `naslib/benchmarks/nas_predictors/discrete_config.yaml` as you desire (e.g., change the search space, predictor, epochs, and set the optimizer to either bananas (Bayesian optimization framework) or npenas (evolution framework)). Then run
```bash
python naslib/benchmarks/nas_predictors/runner.py --config-file naslib/benchmarks/nas_predictors/discrete_config.yaml
```

To run 100 trials of all predictor-based NAS algorithms on some search space and framework (e.g. nas-bench-201 cifar10, evolution), run
```bash
python naslib/benchmarks/predictors/run_nb201_c10_npenas.sh
```

## Citation
Please cite [our paper](https://arxiv.org/abs/2104.01177) if you use code from this repo:

```bibtex
@article{white2021powerful,
  title={How Powerful are Performance Predictors in Neural Architecture Search?},
  author={White, Colin and Zela, Arber and Ru, Binxin and Liu, Yang and Hutter, Frank},
  journal={arXiv preprint arXiv:2104.01177},
  year={2021}
}
```
