#!/bin/bash

experiment=$1

./scripts/correlation/run_nb101.sh $experiment
./scripts/correlation/run_nb201.sh $experiment
./scripts/correlation/run_nb301.sh $experiment
./scripts/correlation/run_tnb101.sh $experiment
