#!/bin/bash

searchspace=nasbench201
datasets=(cifar10 cifar100 ImageNet16-120)
start_seed=9000

experiment=$1
zc_usage=$2
zc_source=$3
n_seeds=$4

if [ -z "$experiment" ]
then
    echo "Experiment argument not provided"
    exit 1
fi

if [ -z "$zc_usage" ]
then
    echo "zc_usage argument not provided"
    exit 1
fi

if [ -z "$zc_source" ]
then
    echo "zc_source argument not provided"
    exit 1
fi

if [ -z "$n_seeds" ]
then
    echo "n_seeds argument not provided"
    exit 1
fi


for dataset in "${datasets[@]}"
do
    echo $searchspace $dataset
    bash ./scripts/zc/zc_ensembles/run.sh $searchspace $dataset $start_seed $n_seeds $experiment $zc_usage $zc_source 
done