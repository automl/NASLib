#!/bin/bash

searchspace=nasbench201
datasets=(cifar10 cifar100 ImageNet16-120)
start_seed=9000

experiment=$1
n_seeds=$2

if [ -z "$experiment" ]
then
    echo "Experiment argument not provided"
    exit 1
fi

for dataset in "${datasets[@]}"
do
    echo $searchspace $dataset
    bash ./scripts/cluster/zc_ensembles/run.sh $searchspace $dataset $start_seed $n_seeds $experiment 
done