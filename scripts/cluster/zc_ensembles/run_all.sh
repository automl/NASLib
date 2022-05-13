#!/bin/bash

experiment=$1

if [ -z "$experiment" ]
then
    echo "Experiment argument not provided"
    exit 1
fi

./scripts/cluster/zc_ensembles/run_nb101.sh $experiment
./scripts/cluster/zc_ensembles/run_nb201.sh $experiment
./scripts/cluster/zc_ensembles/run_nb301.sh $experiment
./scripts/cluster/zc_ensembles/run_tnb101.sh $experiment
