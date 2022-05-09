#!/bin/bash
searchspace=nasbench201
datasets=(cifar10 cifar100 ImageNet16-120)
start_seed=9000
n_seeds=10

for dataset in "${datasets[@]}"
do
    for i in $(seq 0 $(($n_seeds - 1)))
    do
        sbatch ./scripts/cluster/zc_ensembles/run.sh $searchspace $dataset 9000 $(($start_seed + $i))
    done

    echo ""
done
