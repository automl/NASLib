#!/bin/bash

# search space and datasets:
search_space=$1
dataset=$2
start_seed=$3
if [ -z "$start_seed" ]
then
    start_seed=0
fi


out_dir=run
trials=5
end_seed=$(($start_seed + $trials - 1))
test_size=200
config_root=configs

python scripts/create_configs_zc_ensembles.py --start_seed $start_seed --trials $trials --out_dir $out_dir \
    --dataset=$dataset --search_space $search_space --config_root=$config_root --zc_names flops params snip jacov grad_norm plain epe_nas fisher
