#!/bin/bash

train_sizes=(10 15 23 36 56 87 135 209 323 500)
searchspace=transbench101_micro
datasets=(jigsaw class_object class_scene autoencoder)
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