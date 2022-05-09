#!/bin/bash
searchspace=nasbench201
datasets=(cifar10 cifar100 ImageNet16-120)

for dataset in "${datasets[@]}"
do
    sbatch ./scripts/cluster/zc_ensembles/run.sh $searchspace $dataset 9000 5
    echo ""
done