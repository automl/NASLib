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
trials=500
end_seed=$(($start_seed + $trials - 1))
epochs=500
config_root=configs

if [[ "$search_space" == "transbench101_micro"  ||  "$search_space" == "transbench101_macro" ]]; then
    zc_names="flops params snip jacov grad_norm plain fisher grasp l2_norm nwot zen"
else
    zc_names="flops params snip jacov grad_norm plain fisher grasp l2_norm nwot zen epe_nas synflow"
fi

python scripts/create_configs_zc_ensembles.py --start_seed $start_seed --trials $trials --out_dir $out_dir \
    --dataset=$dataset --search_space $search_space --config_root=$config_root --zc_names $zc_names --epochs $epochs
