#!/bin/bash

searchspaces=(transbench101_micro) # transbench101_macro)
datasets=(autoencoder class_object class_scene normal jigsaw room_layout segmentsemantic)
start_seed=9000

experiment=$1
n_seeds=$2

if [ -z "$experiment" ]
then
    echo "Experiment argument not provided"
    exit 1
fi

for searchspace in "${searchspaces[@]}"
do
    for dataset in "${datasets[@]}"
    do
        echo $searchspace $dataset
        bash ./scripts/cluster/zc_ensembles/run.sh $searchspace $dataset $start_seed $n_seeds $experiment 
    done
done
