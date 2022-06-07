#!/bin/bash

searchspace=nasbench201
datasets=(cifar10 cifar100 ImageNet16-120)
start_seed=9000

experiment=$1
n_seeds=$2

if [ -z "$experiment" ]
then
    echo "Experiment argument not provided"
    exit 1
fi

for dataset in "${datasets[@]}"
do
    echo $searchspace $dataset
    sed -i "s/THE_JOB_NAME/${experiment}-${searchspace}-${dataset}/" ./scripts/cluster/zc_ensembles/run.sh
    sbatch ./scripts/cluster/zc_ensembles/run.sh $searchspace $dataset $start_seed $n_seeds $experiment --bosch
    sed -i "s/${experiment}-${searchspace}-${dataset}/THE_JOB_NAME/" ./scripts/cluster/zc_ensembles/run.sh
done