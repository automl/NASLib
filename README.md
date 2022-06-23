
# NAS-Bench-Suite-Zero: Accelerating Research on Zero Cost Proxies

<div align=justify>
`NAS-Bench-Suite-Zero` is a dataset and unified codebase for ZC proxies, enabling orders-of-magnitude faster experiments on ZC proxies, while avoiding confounding factors stemming from different implementations.

`NAS-Bench-Suite-Zero` contains precomputed scores of 13 ZC proxies on 100 to 15625 architectures on 28 tasks, with a total of 1.5M total evaluations. It can be used to run large-scale analyses of ZC proxies, including studies on generalizability and bias of ZC proxies, analyzing mutual information, or integrating ZC proxies into NAS algorithms. 
</div>
  
<div align="center">
  <img src="images/nas-bench-suite-zero.png" width="1000" height="377">
</div>


[**Setup**](#setup)
| [**Tutorial**](#tutorial)
| [**Usage**](#usage)


# Setup

While installing the repository, creating a new conda environment is recomended. [Install PyTorch GPU/CPU](https://pytorch.org/get-started/locally/) for your setup.

```bash
git clone -b automl-conf-competition https://github.com/automl/NASLib/
cd NASLib
conda create -n automl-competition  python=3.9
conda activate automl-competition
```

Run setup.py file with the following command, which will install all the packages listed in [`requirements.txt`](requirements.txt).
```bash
pip install --upgrade pip setuptools wheel
pip install -e .
```
Download all the datasets and benchmarks ( for mac users, please make sure you have wget installed).
```bash
source scripts/bash_scripts/download_data.sh all 
```
Alternatively, you can download the benchmark for a specific search space and dataset/task as follows:
```bash
source scripts/bash_scripts/download_data.sh <search_space> <dataset> 
source scripts/bash_scripts/download_data.sh nb201 cifar10
source scripts/bash_scripts/download_data.sh nb201 all 
```
Download the TransNAS-Bench-101 benchmark from [here](https://www.noahlab.com.hk/opensource/vega/page/doc.html?path=datasets/transnasbench101) unzip the folder and place the benchmark `transnas-bench_v10141024.pth` from this folder in `NASLib/naslib/data/..`

If you face issues downloading the datasets please follow the steps [here](dataset_preparation/).

# Tutorial
This [tutorial](tutorial/) will help participants get acquainted with NASLib and a sample submission.

# Usage
To test the setup on different benchmarks you can run

```bash
bash scripts/bash_scripts/run_nb201.sh
bash scripts/bash_scripts/run_nb301.sh
bash scripts/bash_scripts/run_tnb101.sh
```

