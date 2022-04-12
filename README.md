
<div align="center">
  <img src="images/zcp_competition.png" width="1000" height="200">
</div>
<div>

</div>

<div align=justify>

This is the official github repo for the [Zero-Cost NAS Competition](https://sites.google.com/view/zero-cost-nas-competition/home) organized at the [AutoML-Conf 22](https://automl.cc/). This competition shall be hosted on CodaLab. Participants will be required to implement their zero-cost proxies using a lightweight version of [NASLib](https://github.com/automl/NASLib), a library for Neural Architecture Search. NASLib provides users with a range of tabular and surrogate benchmarks, making it easy to sample a random architecture from a supported search space, instantiate it as a PyTorch model, and query its final performance instantly. Once a zero-cost proxy has been implemented, the framework allows users to evaluate its performance across several search spaces and tasks in a matter of minutes.

</div>

<div align=justify>

The challenge is as follows: Given N models from a search space, such as [NASBench301](https://arxiv.org/pdf/2008.09777.pdf), the participant's zero-cost proxy will be used to score and rank the models for a given task, such as classification on CIFAR10 dataset. The Kendall-Tau rank correlation between the predicted and actual ranks of the models is the metric of interest. The final score of a submission shall be the average rank correlation across a set of NAS benchmarks (combinations of search spaces and datasets). To keep the spirit of "zero-cost" proxies in the user submissions, it is required that the scoring of models consumes only negligible computational resources. This is enforced by running the computations on CPUs instead of GPUs and setting a hard limit for the runtime of the program.

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

