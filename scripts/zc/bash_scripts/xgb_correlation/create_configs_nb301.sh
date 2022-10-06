#!/bin/bash

experiment=$1

if [ -z "$experiment" ]
then
    echo Experiment argument not provided
    exit 1
fi

searchspace=nasbench301
datasets=(cifar10)

ks=(1 2 3 4 5 6 7 8 9 10 11 12 13)
proxies=(
"nwot"
"nwot epe_nas"
"nwot epe_nas synflow"
"nwot epe_nas synflow plain"
"nwot epe_nas synflow plain grad_norm"
"nwot epe_nas synflow plain grad_norm zen"
"nwot epe_nas synflow plain grad_norm zen l2_norm"
"nwot epe_nas synflow plain grad_norm zen l2_norm flops"
"nwot epe_nas synflow plain grad_norm zen l2_norm flops snip"
"nwot epe_nas synflow plain grad_norm zen l2_norm flops snip fisher"
"nwot epe_nas synflow plain grad_norm zen l2_norm flops snip fisher params"
"nwot epe_nas synflow plain grad_norm zen l2_norm flops snip fisher params jacov"
"nwot epe_nas synflow plain grad_norm zen l2_norm flops snip fisher params jacov grasp"
)

for i in "${!proxies[@]}"
do
    echo "${proxies[$i]}"
    for dataset in "${datasets[@]}"
    do
        scripts/zc/bash_scripts/xgb_correlation/create_configs.sh $experiment $searchspace $dataset 9000 "${ks[$i]}" "${proxies[$i]}"
    done
done