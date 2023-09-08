# Setup

While installing the repository, creating a new conda environment is recomended. [Install PyTorch GPU/CPU](https://pytorch.org/get-started/locally/) for your setup.

```bash
conda create -n mvenv python=3.7
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
```

Run setup.py file with the following command, which will install all the packages listed in `requirements.txt`.
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