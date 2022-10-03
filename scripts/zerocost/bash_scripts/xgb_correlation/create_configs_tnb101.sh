#!/bin/bash

experiment=$1

if [ -z "$experiment" ]
then
    echo Experiment argument not provided
    exit 1
fi

searchspaces=(transbench101_micro transbench101_macro)
datasets=(autoencoder class_object class_scene normal jigsaw room_layout segmentsemantic)

for searchspace in "${searchspaces[@]}"
do
    for dataset in "${datasets[@]}"
    do
        scripts/bash_scripts/xgb_correlation/create_configs.sh $experiment $searchspace $dataset 9000
    done
done