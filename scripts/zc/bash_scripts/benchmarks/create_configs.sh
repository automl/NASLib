#!/bin/bash

# search space and datasets:
search_space=$1
dataset=$2
start_seed=$3
if [ -z "$start_seed" ]
then
    start_seed=0
fi

# folders:
config_root='naslib/configs'
out_dir=run

# predictors
predictors=(fisher grad_norm grasp jacov snip synflow epe_nas flops params plain l2_norm nwot zen)

# other variables:
trials=10
end_seed=$(($start_seed + $trials - 1))
train_size=100

# create config files
for i in $(seq 0 $((${#predictors[@]}-1)) )
  do
    predictor=${predictors[$i]}
    python scripts/zc/create_configs_benchmarks.py --predictor $predictor \
    --start_seed $start_seed --out_dir $out_dir --dataset=$dataset \
    --experiment benchmarks --search_space $search_space --config_root=$config_root
done