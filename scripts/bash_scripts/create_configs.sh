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
base_file=naslib
config_folder=configs/predictors/${search_space}
config_root=$config_folder-$start_seed
out_dir=run

# predictors
predictors=(fisher grad_norm grasp jacov snip synflow epe_nas flops params)

# other variables:
trials=5
end_seed=$(($start_seed + $trials - 1))
test_size=200

# create config files
for i in $(seq 0 $((${#predictors[@]}-1)) )
do
    predictor=${predictors[$i]}
    python scripts/create_configs.py --predictor $predictor \
    --test_size $test_size --start_seed $start_seed --trials $trials --out_dir $out_dir \
    --dataset=$dataset --config_type predictor --search_space $search_space \
    --config_root=$config_root
done
