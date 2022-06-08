#!/bin/bash

experiment=$1

if [ -z "$experiment" ]
then
    echo Experiment argument not provided
    exit 1
fi

searchspace=nasbench301
datasets=(cifar10)

for dataset in "${datasets[@]}"
do
    scripts/bash_scripts/xgb_correlation/create_configs.sh $experiment $searchspace $dataset 9000
done