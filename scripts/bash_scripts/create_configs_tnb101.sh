#!/bin/bash

searchspace=transbench101_micro
datasets=(autoencoder class_object class_scene normal jigsaw room_layout segmentsemantic)

for dataset in "${datasets[@]}"
do
    scripts/bash_scripts/create_configs.sh $searchspace $dataset 9000
done