#!/bin/bash

# train_sizes=(5 8 14 24 42 71 121 205 347 589 1000) #(1000) #(10 15 23 36 56 87 135 209 323 500)
train_sizes=(1000)
searchspace=nasbench201
# datasets=(cifar10 cifar100 ImageNet16-120)
ks=(1 2 3 4 5 6 7 8 9 10 11 12 13)
datasets=(cifar100)
start_seed=9000

experiment=$1

if [ -z "$experiment" ]
then
    echo "Experiment argument not provided"
    exit 1
fi

for k in "${ks[@]}"
do
    for dataset in "${datasets[@]}"
    do
        for size in "${train_sizes[@]}"
        do
            sbatch ./scripts/zc/xgb_correlation/run.sh $searchspace $dataset $size $start_seed $experiment $k --bosch
        done

    done
done
