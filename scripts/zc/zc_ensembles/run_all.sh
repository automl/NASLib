#!/bin/bash

experiment=$1

n_seeds=10

if [ -z "$experiment" ]
then
    echo "Experiment argument not provided"
    exit 1
fi

# ./scripts/zc/zc_ensembles/run_nb101.sh $experiment $n_seeds
./scripts/zc/zc_ensembles/run_nb201.sh $experiment $n_seeds
./scripts/zc/zc_ensembles/run_nb301.sh $experiment $n_seeds
./scripts/zc/zc_ensembles/run_tnb101.sh $experiment $n_seeds
