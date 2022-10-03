#!/bin/bash

experiment=$1

./scripts/bash_scripts/xgb_correlation/create_configs_nb101.sh $experiment
./scripts/bash_scripts/xgb_correlation/create_configs_nb201.sh $experiment
./scripts/bash_scripts/xgb_correlation/create_configs_nb301.sh $experiment
./scripts/bash_scripts/xgb_correlation/create_configs_tnb101.sh $experiment
