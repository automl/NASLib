#!/bin/bash

predictor=$1

if [ -z "$predictor" ];
then
    predictors=(fisher grad_norm grasp jacov snip synflow epe_nas flops params plain l2_norm nwot)
else
    predictors=($predictor)
fi

searchspace=nasbench101
datasets=(cifar10)

for dataset in "${datasets[@]}"
do
    for pred in "${predictors[@]}"
    do
        sbatch ./scripts/cluster/correlation/run.sh $searchspace $dataset $pred 9000
    done

    echo ""
done