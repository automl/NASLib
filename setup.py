import os
import sys
import subprocess
from setuptools import setup, find_packages

VERBOSE_SCRIPT = True
REQUIRED_MAJOR = 3
REQUIRED_MINOR = 5

# Check for python version
if sys.version_info < (REQUIRED_MAJOR, REQUIRED_MINOR):
    error = (
        "Your version of python ({major}.{minor}) is too old. You need "
        "python >= {required_major}.{required_minor}."
    ).format(
        major=sys.version_info.major,
        minor=sys.version_info.minor,
        required_minor=REQUIRED_MINOR,
        required_major=REQUIRED_MAJOR,
    )
    sys.exit(error)

cwd = os.path.dirname(os.path.abspath(__file__))
version = open('.version', 'r').read().strip()

if VERBOSE_SCRIPT:
    def report(*args):
        print(*args)
else:
    def report(*args):
        pass

version_path = os.path.join(cwd, 'naslib', '__init__.py')
with open(version_path, 'w') as f:
    report('-- Building version ' + version)
    f.write("__version__ = '{}'\n".format(version))

requires = [
    "ConfigSpace",
    "cython",
    "george==0.3.1",
    "hyperopt==0.1.2",
    "pyyaml",
    "cave",
    "hpbandster",
    "numpy==1.16.4",
    "pyrfr==0.8.0",
    "Pyro4==4.76",
    "RoBO==0.3.1",
    "torch>=1.2.0",
    "torchvision>=0.4.0",
    "tensorflow-gpu==1.14.0",
    "tensorflow-probability==0.7.0",
]

import subprocess

git_nasbench = "git+https://github.com/google-research/nasbench.git@master"

try:
    import nasbench
except ImportError:
    if '--user' in sys.argv:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade',
            '--user', git_nasbench], check=False)
    else:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade',
            git_nasbench], check=False)


if __name__=='__main__':
    setup(
        name='naslib',
        version=version,
        description='NASLib: A Neural Architecture Search (NAS) library.',
        author='AutoML Freiburg',
        author_email='zelaa@cs.uni-freiburg.de',
        url='https://github.com/automl/NASLib',
        license='Apache License 2.0',
        classifiers=['Development Status :: 1 - Beta'],
        packages=find_packages(),
        install_requires=requires,
        keywords=['NAS', 'automl'],
        #test_suite='tests'
    )
