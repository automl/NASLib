<div align="center">
  ** For the <a href='https://codalab.lisn.upsaclay.fr/competitions/3932'>Zero-Cost NAS Competition</a>, please switch to the <a href='https://github.com/automl/NASLib/tree/automl-conf-competition'><code>automl-conf-competition</code></a> branch ** <br><br>

  <img src="images/naslib-logo.png" width="400" height="250">
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

<p align="center">
  <img src="https://repobeats.axiom.co/api/embed/44112451b6b665f03b7c7dc35dbeff8050df036f.svg" width="750" />
</p>



&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
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

## Queryable Benchmarks
NASLib allows you to query the following (tabular and surrogate) benchmarks for the performance of any architecture, for a given search space, dataset and task. To set them up, simply download the benchmark data files from the these URLs and place them in `naslib/data`.

| Benchmark     | Task                               | Datasets |        Data URL       | Required Files |
|---------------|------------------------------------|----------|-----------------------|----------------|
|NAS-Bench-101  | Image Classification   |                    CIFAR10                   | [cifar10](https://drive.google.com/file/d/1oORtEmzyfG1GcnPHh0ijCs0gCHKEThNx/view?usp=sharing)| `naslib/data/nasbench_only108.pkl` |
|NAS-Bench-201  | Image Classification   |  CIFAR10 <br> CIFAR100 <br> ImageNet16-120   | [cifar10](https://drive.google.com/file/d/1sh8pEhdrgZ97-VFBVL94rI36gedExVgJ/view?usp=sharing) <br> [cifar100](https://drive.google.com/file/d/1hV6-mCUKInIK1iqZ0jfBkcKaFmftlBtp/view?usp=sharing) <br> [imagenet](https://drive.google.com/file/d/1FVCn54aQwD6X6NazaIZ_yjhj47mOGdIH/view?usp=sharing)| `naslib/data/nb201_cifar10_full_training.pickle` <br> `naslib/data/nb201_cifar100_full_training.pickle` <br>  `naslib/data/nb201_ImageNet16_full_training.pickle`|
|NAS-Bench-301  | Image Classification   |                    CIFAR10                   |  [cifar10](https://drive.google.com/file/d/1YJ80Twt9g8Gaf8mMgzK-f5hWaVFPlECF/view?usp=sharing)<br> [models](https://figshare.com/articles/software/nasbench301_models_v1_0_zip/13061510) |`naslib/data/nb301_full_training.pickle` <br> `naslib/data/nb_models/...`|
|NAS-Bench-ASR  | Automatic Speech Recognition  |               TIMIT                   |  [timit](https://github.com/SamsungLabs/nb-asr/releases/tag/v1.1.0) | `naslib/data/nb-asr-bench-gtx-1080ti-fp32.pickle` <br> `naslib/data/nb-asr-bench-jetson-nano-fp32.pickle` <br> `naslib/data/nb-asr-e40-1234.pickle` <br> `naslib/data/nb-asr-e40-1235.pickle` <br> `naslib/data/nb-asr-e40-1236.pickle` <br> `naslib/data/nb-asr-info.pickle`
|NAS-Bench-NLP  | Natural Language Processing   |           Penn Treebank               |               [ptb](https://drive.google.com/file/d/1DtrmuDODeV2w5kGcmcHcGj5JXf2qWg01/view?usp=sharing), [models](https://drive.google.com/file/d/13Kbn9VWHuBdSN3lG4Mbyr2-VdrTsfLfd/view?usp=sharing)| `naslib/data/nb_nlp.pickle` <br> `naslib/data/nbnlp_v01/...`|
|TransNAS-Bench-101  | 7 Computer Vision tasks  |             Taskonomy                 | [taskonomy](https://www.noahlab.com.hk/opensource/vega/page/doc.html?path=datasets/transnasbench101) |`naslib/data/transnas-bench_v10141024.pth`|

For `NAS-Bench-301` and `NAS-Bench-NLP`, additionally, you will have to install the NASBench301 API from [here](https://github.com/crwhite14/nasbench301).

Once set up, you can test if the APIs work as follows:
```
python test_benchmark_apis.py --all --show_error
```
You can also test any one API.
```
python test_benchmark_apis.py --search_space <search_space> --show_error
```
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

### Scripts for running multiple experiments on a cluster
The `scripts` folder contains code for generating config files for running experiments across various configurations and seeds. It writes them into the `naslib/configs` folder. 

```bash
cd scripts
bash bbo/make_configs_asr.sh
```

It also contains `scheduler.sh` files to automatically read these generated config files and submits a corresponding job to the cluster using SLURM.

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
  
@inproceedings{mehta2022bench,
  title={NAS-Bench-Suite: NAS Evaluation is (Now) Surprisingly Easy},
  author={Mehta, Yash and White, Colin and Zela, Arber and Krishnakumar, Arjun and Zabergja, Guri and Moradian, Shakiba and Safari, Mahmoud and Yu, Kaicheng and Hutter, Frank},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
 ``` 
 

<p align="center">
  <img src="images/predictors.png" alt="predictors" width="75%">
</p>

NASLib has been used to run an extensive comparison of 31 performance predictors (figure above). See the separate readme: <a href="examples/predictors.md">predictors.md</a>
and our paper: <a href="https://arxiv.org/abs/2104.01177">How Powerful are Performance Predictors in Neural Architecture Search?</a>

```bibtex
@article{white2021powerful,
  title={How Powerful are Performance Predictors in Neural Architecture Search?},
  author={White, Colin and Zela, Arber and Ru, Robin and Liu, Yang and Hutter, Frank},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```
