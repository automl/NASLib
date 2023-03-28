#!/bin/bash

experiment=$1
zc_usage=$2
zc_source=$3
n_seeds=10

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

bash scripts/zc/zc_ensembles/run_nb101.sh $experiment $zc_usage $zc_source $n_seeds
bash scripts/zc/zc_ensembles/run_nb201.sh $experiment $zc_usage $zc_source $n_seeds
bash scripts/zc/zc_ensembles/run_nb301.sh $experiment $zc_usage $zc_source $n_seeds
bash scripts/zc/zc_ensembles/run_tnb101.sh $experiment $zc_usage $zc_source $n_seeds
