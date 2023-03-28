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
"flops"
"flops l2_norm"
"flops l2_norm plain"
"flops l2_norm plain jacov"
"flops l2_norm plain jacov grad_norm"
"flops l2_norm plain jacov grad_norm nwot"
"flops l2_norm plain jacov grad_norm nwot params"
"flops l2_norm plain jacov grad_norm nwot params fisher"
"flops l2_norm plain jacov grad_norm nwot params fisher zen"
"flops l2_norm plain jacov grad_norm nwot params fisher zen snip"
"flops l2_norm plain jacov grad_norm nwot params fisher zen snip grasp"
)

for i in "${!proxies[@]}"
do
    echo "${proxies[$i]}"
    for searchspace in "${searchspaces[@]}"
    do
        for dataset in "${datasets[@]}"
        do
            scripts/zc/bash_scripts/xgb_correlation/create_configs.sh $experiment $searchspace $dataset 9000 "${ks[$i]}" "${proxies[$i]}"
        done
    done
done