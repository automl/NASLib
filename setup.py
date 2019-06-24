import os
import setuptools

requirements = []
with open('requirements.txt', 'r') as f:
    for line in f:
        requirements.append(line.strip())


setuptools.setup(
    name="naslib",
    version="0.0.1",
    author="AutoML Freiburg",
    author_email="zimmerl@informatik.uni-freiburg.de",
    description=("NASOpt provides optimizers for neural architecture search"),
    license="3-clause BSD",
    keywords="machine learning algorithm configuration hyperparameter "
             "optimization tuning neural architecture search deep learning",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: 3-clause BSD",
    ],
	python_requires='>=3',
    platforms=['Linux'],
    install_requires=requirements
)
