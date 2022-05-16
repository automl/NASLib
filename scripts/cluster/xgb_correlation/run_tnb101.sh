#!/bin/bash

train_sizes=(100)
searchspaces=(transbench101_micro transbench101_macro)
datasets=(class_object class_scene room_layout)
# datasets=(autoencoder normal segmentsemantic)
start_seed=9000
n_seeds=10

experiment=$1

if [ -z "$experiment" ]
then
    echo "Experiment argument not provided"
    exit 1
fi

for searchspace in "${searchspaces[@]}"
do
    for dataset in "${datasets[@]}"
    do
        for size in "${train_sizes[@]}"
        do
            sbatch ./scripts/cluster/xgb_correlation/run.sh $searchspace $dataset $size $start_seed $experiment <<< "y"
        done

        # echo ""
    done
done
