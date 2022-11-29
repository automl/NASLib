import os
import sys
import subprocess
from setuptools import setup, find_packages

# Check for python version
if sys.version_info.major != 3 or sys.version_info.minor < 7 or sys.version_info.minor > 9:
    raise ValueError(
        'Unsupported Python version %d.%d.%d found. NASLib requires Python '
        '3.7, 3.8 or 3.9' % (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
    )


cwd = os.path.dirname(os.path.abspath(__file__))

version_path = os.path.join(cwd, 'naslib', '__version__.py')
with open(version_path) as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")

with open("README.md", "r") as f:
    long_description = f.read()

requirements = []
with open("requirements.txt", "r") as f:
    for line in f:
        requirements.append(line.strip())

git_nb301 = "git+https://github.com/automl/nasbench301.git@no_gin"
git_archai = "git+https://github.com/microsoft/archai.git@a46590cebd057c03b1d2591bf3e0bb4454ff0470"

try:
    import nasbench301
except ImportError:
    if '--user' in sys.argv:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade',
            '--user', git_nb301], check=False)
    else:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade',
            git_nb301], check=False)

try:
    import archai
except ImportError:
    if '--user' in sys.argv:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade',
            '--user', git_archai], check=False)
    else:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade',
            git_archai], check=False)

print('-- Building version ' + version)
print('-- Note: by default installs pytorch-cpu version (1.9.0), update to torch-gpu by following instructions from: https://pytorch.org/get-started/locally/')

setup(
    name='naslib',
    version=version,
    description='NASLib: A modular and extensible Neural Architecture Search (NAS) library.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='AutoML Freiburg',
    author_email='zelaa@cs.uni-freiburg.de',
    url='https://github.com/automl/NASLib',
    license='Apache License 2.0',
    classifiers=['Development Status :: 1 - Beta'],
    packages=find_packages(),
    python_requires='>=3.7',
    platforms=['Linux'],
    install_requires=requirements,
    keywords=['NAS', 'automl'],
    test_suite='pytest'
)
