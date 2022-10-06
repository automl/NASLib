#!/bin/bash

experiment=$1

if [ -z "$experiment" ]
then
    echo Experiment argument not provided
    exit 1
fi

searchspace=nasbench101
datasets=(cifar10)

ks=(1 2 3 4 5 6 7 8 9 10 11 12 13)
proxies=(
"zen"
"zen epe_nas"
"zen epe_nas jacov"
"zen epe_nas jacov synflow"
"zen epe_nas jacov synflow plain"
"zen epe_nas jacov synflow plain nwot"
"zen epe_nas jacov synflow plain nwot grad_norm"
"zen epe_nas jacov synflow plain nwot grad_norm l2_norm"
"zen epe_nas jacov synflow plain nwot grad_norm l2_norm grasp"
"zen epe_nas jacov synflow plain nwot grad_norm l2_norm grasp fisher"
"zen epe_nas jacov synflow plain nwot grad_norm l2_norm grasp fisher snip"
"zen epe_nas jacov synflow plain nwot grad_norm l2_norm grasp fisher snip params"
"zen epe_nas jacov synflow plain nwot grad_norm l2_norm grasp fisher snip params flops"
)

for i in "${!proxies[@]}"
do
    echo "${proxies[$i]}"
    for dataset in "${datasets[@]}"
    do
        scripts/zc/bash_scripts/xgb_correlation/create_configs.sh $experiment $searchspace $dataset 9000 "${ks[$i]}" "${proxies[$i]}"
    done
done