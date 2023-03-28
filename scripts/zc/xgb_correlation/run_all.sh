#!/bin/bash

experiment=$1

if [ -z "$experiment" ]
then
    echo "Experiment argument not provided"
    exit 1
fi

./scripts/zc/xgb_correlation/run_nb101.sh $experiment
./scripts/zc/xgb_correlation/run_nb201.sh $experiment
./scripts/zc/xgb_correlation/run_nb301.sh $experiment
./scripts/zc/xgb_correlation/run_tnb101.sh $experiment
