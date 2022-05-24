#!/bin/bash

searchspaces=(transbench101_micro transbench101_macro)
datasets=(autoencoder class_object class_scene normal jigsaw room_layout segmentsemantic)
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
        for i in $(seq 0 $(($n_seeds - 1)))
        do
            sbatch ./scripts/cluster/zc_ensembles/run.sh $searchspace $dataset $start_seed $(($start_seed + $i)) $experiment <<< "y"
        done

        echo ""
    done
done
