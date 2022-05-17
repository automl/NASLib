#!/bin/bash

experiment=benchmarks

./scripts/cluster/sampler/run_nb101.sh $experiment
./scripts/cluster/sampler/run_nb201.sh $experiment
./scripts/cluster/sampler/run_nb301.sh $experiment
./scripts/cluster/sampler/run_tnb101.sh $experiment
