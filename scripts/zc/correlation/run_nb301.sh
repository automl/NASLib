#!/bin/bash

experiment=$1
predictor=$2
start_seed=9000

if [ -z "$experiment" ]
then
    echo "Experiment argument not provided"
    exit 1
fi

if [ -z "$predictor" ];
then
    predictors=(fisher grad_norm grasp jacov snip synflow epe_nas flops params plain l2_norm nwot zen)
else
    predictors=($predictor)
fi

searchspace=nasbench301
datasets=(cifar10)

for dataset in "${datasets[@]}"
do
    for pred in "${predictors[@]}"
    do
        sed -i "s/THE_JOB_NAME/${searchspace}-${dataset}-${pred}/" ./scripts/zc/correlation/run.sh
        echo $searchspace $dataset $pred
        sbatch ./scripts/zc/correlation/run.sh $searchspace $dataset $pred $start_seed $experiment --bosch
        sed -i "s/${searchspace}-${dataset}-${pred}/THE_JOB_NAME/" ./scripts/zc/correlation/run.sh
    done

    echo ""
done