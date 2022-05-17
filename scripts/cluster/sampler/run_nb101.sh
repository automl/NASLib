#!/bin/bash

experiment=$1
start_seed=9000

if [ -z "$experiment" ]
then
    echo "Experiment argument not provided"
    exit 1
fi

searchspace=nasbench101
datasets=(cifar10)

for dataset in "${datasets[@]}"
do
    sbatch ./scripts/cluster/sampler/run.sh $searchspace $dataset $start_seed $experiment <<< "y"
done