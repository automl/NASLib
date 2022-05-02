#!/bin/bash

searchspace=transbench101
datasets=(autoencoder class_object class_scene normal jigsaw room_layout segmentsemantic)

for dataset in "${datasets[@]}"
do
    naslib/benchmarks/predictors/create_configs.sh $searchspace $dataset 9000
done