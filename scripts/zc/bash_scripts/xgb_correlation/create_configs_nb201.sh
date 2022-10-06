#!/bin/bash

experiment=$1

if [ -z "$experiment" ]
then
    echo Experiment argument not provided
    exit 1
fi

searchspace=nasbench201
datasets=(cifar100) #cifar10 cifar100 ImageNet16-120)

ks=(1 2 3 4 5 6 7 8 9 10 11 12 13)
proxies=(
"synflow"
"synflow plain"
"synflow plain l2_norm"
"synflow plain l2_norm flops"
"synflow plain l2_norm flops snip"
"synflow plain l2_norm flops snip grad_norm"
"synflow plain l2_norm flops snip grad_norm nwot"
"synflow plain l2_norm flops snip grad_norm nwot zen"
"synflow plain l2_norm flops snip grad_norm nwot zen fisher"
"synflow plain l2_norm flops snip grad_norm nwot zen fisher jacov"
"synflow plain l2_norm flops snip grad_norm nwot zen fisher jacov epe_nas"
"synflow plain l2_norm flops snip grad_norm nwot zen fisher jacov epe_nas params"
"synflow plain l2_norm flops snip grad_norm nwot zen fisher jacov epe_nas params grasp"
)

for i in "${!proxies[@]}"
do
    echo "${proxies[$i]}"
    for dataset in "${datasets[@]}"
    do
        scripts/zc/bash_scripts/xgb_correlation/create_configs.sh $experiment $searchspace $dataset 9000 "${ks[$i]}" "${proxies[$i]}"
    done
done