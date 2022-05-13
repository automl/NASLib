#!/bin/bash

experiment=$1
predictor=$2
start_seed=9000

if [ -z "$experiment" ]
then
    echo "Experiment argument not provided"
    exit 1
fi

if [ -z "$predictor" ];
then
    predictors=(fisher grad_norm grasp jacov snip synflow epe_nas flops params plain l2_norm nwot)
else
    predictors=($predictor)
fi

searchspaces=(transbench101_micro transbench101_macro)
datasets=(autoencoder class_object class_scene normal jigsaw room_layout segmentsemantic)


for searchspace in "${searchspaces[@]}"
do
    for dataset in "${datasets[@]}"
    do
        for pred in "${predictors[@]}"
        do
            sbatch ./scripts/cluster/correlation/run.sh $searchspace $dataset $pred $start_seed $experiment <<< "y"
        done

        echo ""
    done
done