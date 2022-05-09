#!/bin/bash
searchspace=nasbench301
datasets=(cifar10)

for dataset in "${datasets[@]}"
do
    sbatch ./scripts/cluster/zc_ensembles/run.sh $searchspace $dataset 9000 5
    echo ""
done