#!/bin/bash

# search space and datasets:
experiment=$1
search_space=$2
dataset=$3
start_seed=$4
k=$5
zc_names=$6

if [ -z "$experiment" ]
then
    echo Experiment argument not provided
    exit 1
fi

if [ -z "$search_space" ]
then
    echo Search space argument not provided
    exit 1
fi

if [ -z "$dataset" ]
then
    echo Dataset argument not provided
    exit 1
fi

if [ -z "$start_seed" ]
then
    start_seed=0
fi

out_dir=run
trials=100
end_seed=$(($start_seed + $trials - 1))
# train_sizes=(5 8 14 24 42 71 121 205 347 589 1000)
train_sizes=(1000)
test_size=1000
config_root='naslib/configs'

# if [[ "$search_space" == "transbench101_micro"  ||  "$search_space" == "transbench101_macro" ]]; then
#     zc_names="flops params snip jacov grad_norm plain fisher grasp l2_norm nwot zen"
# else
#     zc_names="flops params snip jacov grad_norm plain fisher grasp l2_norm nwot zen epe_nas synflow"
# fi

if [[ "$experiment" == "xgb_only_zc" ]]; then
    echo xgb_only_zc
    zc_ensemble=True
    zc_only=True
fi

if [[ "$experiment" == "xgb_only_adjacency" ]]; then
    echo xgb_only_adjacency
    zc_ensemble="False"
    zc_only="False"
fi

echo bash $zc_ensemble $zc_only

if [ -z "$zc_ensemble" ]
then
    echo zc_ensemble not set
    exit 1
fi

if [ -z "$zc_only" ]
then
    echo zc_only not set
    exit 1
fi

for train_size in "${train_sizes[@]}"
do

python scripts/zc/create_configs_xgb_correlation.py --start_seed $start_seed --trials $trials --out_dir $out_dir \
    --dataset=$dataset --search_space $search_space --config_root=$config_root --zc_names $zc_names \
    --train_size $train_size --experiment $experiment --zc_ensemble $zc_ensemble --zc_only $zc_only \
    --test_size $test_size --k $k
done