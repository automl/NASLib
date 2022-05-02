#!/bin/bash

searchspace=nasbench201
datasets=(cifar10 cifar100 ImageNet16-120)

for dataset in "${datasets[@]}"
do
    naslib/benchmarks/predictors/create_configs.sh $searchspace $dataset 9000
done