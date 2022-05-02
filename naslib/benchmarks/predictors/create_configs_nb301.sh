#!/bin/bash

searchspace=nasbench301
datasets=(cifar10)

for dataset in "${datasets[@]}"
do
    naslib/benchmarks/predictors/create_configs.sh $searchspace $dataset 9000
done