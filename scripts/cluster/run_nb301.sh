#!/bin/bash

predictor=$1

if [ -z "$predictor" ];
then
    predictors=(params fisher grad_norm grasp jacov snip synflow epe_nas flops)
else
    predictors=($predictor)
fi

searchspace=nasbench301
datasets=(cifar10)

for dataset in "${datasets[@]}"
do
    for pred in "${predictors[@]}"
    do
        sbatch ./scripts/cluster/run.sh $searchspace $dataset $pred 9000 5
    done

    echo ""
done