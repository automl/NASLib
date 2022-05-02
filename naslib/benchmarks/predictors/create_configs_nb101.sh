#!/bin/bash

searchspace=nasbench101
datasets=(cifar10)

for dataset in "${datasets[@]}"
do
    naslib/benchmarks/predictors/create_configs.sh $searchspace $dataset 9000
done