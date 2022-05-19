#!/bin/bash

train_sizes=(400)
searchspace=nasbench301
datasets=(cifar10)
start_seed=9000
n_seeds=10

experiment=$1

if [ -z "$experiment" ]
then
    echo "Experiment argument not provided"
    exit 1
fi

for dataset in "${datasets[@]}"
do
    for size in "${train_sizes[@]}"
    do
        sbatch ./scripts/cluster/xgb_correlation/run.sh $searchspace $dataset $size $start_seed $experiment <<< "y"
    done

    echo ""
done