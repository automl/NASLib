#!/bin/bash

experiment=$1

./scripts/zc/benchmarks/run_nb101.sh $experiment
./scripts/zc/benchmarks/run_nb201.sh $experiment
./scripts/zc/benchmarks/run_nb301.sh $experiment
./scripts/zc/benchmarks/run_tnb101.sh $experiment
