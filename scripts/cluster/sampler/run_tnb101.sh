#!/bin/bash

experiment=$1
start_seed=9000

if [ -z "$experiment" ]
then
    echo "Experiment argument not provided"
    exit 1
fi

searchspaces=(transbench101_micro transbench101_macro)
datasets=(autoencoder class_object class_scene normal jigsaw room_layout segmentsemantic)


for searchspace in "${searchspaces[@]}"
do
    for dataset in "${datasets[@]}"
    do
        sbatch ./scripts/cluster/sampler/run.sh $searchspace $dataset $start_seed $experiment <<< "y"
    done
done