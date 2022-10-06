#!/bin/bash

searchspace=nasbench201
datasets=(cifar10 cifar100 ImageNet16-120 svhn scifar100 ninapro)

for dataset in "${datasets[@]}"
do
    scripts/zc/bash_scripts/correlation/create_configs.sh $searchspace $dataset 9000
done
