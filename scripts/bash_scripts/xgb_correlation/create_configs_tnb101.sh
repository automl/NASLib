#!/bin/bash

experiment=$1

if [ -z "$experiment" ]
then
    echo Experiment argument not provided
    exit 1
fi

searchspaces=(transbench101_macro) # transbench101_micro)
datasets=(autoencoder) # class_object class_scene normal jigsaw room_layout segmentsemantic)

ks=(1 2 3 4 5 6 7 8 9 10 11)
proxies=(
"l2_norm"
"l2_norm plain"
"l2_norm plain params"
"l2_norm plain params snip"
"l2_norm plain params snip nwot"
"l2_norm plain params snip nwot grad_norm"
"l2_norm plain params snip nwot grad_norm flops"
"l2_norm plain params snip nwot grad_norm flops grasp"
"l2_norm plain params snip nwot grad_norm flops grasp fisher"
"l2_norm plain params snip nwot grad_norm flops grasp fisher zen"
"l2_norm plain params snip nwot grad_norm flops grasp fisher zen jacov"
)

for i in "${!proxies[@]}"
do
    echo "${proxies[$i]}"
    for searchspace in "${searchspaces[@]}"
    do
        for dataset in "${datasets[@]}"
        do
            scripts/bash_scripts/xgb_correlation/create_configs.sh $experiment $searchspace $dataset 9000 "${ks[$i]}" "${proxies[$i]}"
        done
    done
done