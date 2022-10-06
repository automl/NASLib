#!/bin/bash

searchspace=nasbench301
datasets=(cifar10)

for dataset in "${datasets[@]}"
do
    scripts/zc/bash_scripts/zc_ensemble/create_configs.sh $searchspace $dataset 9000
done