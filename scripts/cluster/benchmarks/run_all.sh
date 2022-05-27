#!/bin/bash

experiment=$1

./scripts/cluster/benchmarks/run_nb101.sh $experiment
./scripts/cluster/benchmarks/run_nb201.sh $experiment
./scripts/cluster/benchmarks/run_nb301.sh $experiment
./scripts/cluster/benchmarks/run_tnb101.sh $experiment
