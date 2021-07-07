import os
import sys
import subprocess
from setuptools import setup, find_packages

VERBOSE_SCRIPT = True

# Check for python version
if sys.version_info < (3, 7):
    raise ValueError(
        'Unsupported Python version %d.%d.%d found. NASLib requires Python '
        '3.7 or higher.' % (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
    )

cwd = os.path.dirname(os.path.abspath(__file__))

with open("naslib/__version__.py") as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")

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

with open("README.md", "r") as f:
    long_description = f.read()

requirements = []
with open("requirements.txt", "r") as f:
    for line in f:
        requirements.append(line.strip())

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
