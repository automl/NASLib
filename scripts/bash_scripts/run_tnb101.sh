predictors=(jacov2 snip synflow fisher grad_norm grasp)
experiment_types=(single single single single single single)

start_seed=$1
if [ -z "$start_seed" ]
then
    start_seed=0
fi

# folders:
base_file=scripts
s3_folder=class_scene_zc_dec10_2021
out_dir=$s3_folder\_$start_seed

# search space / data:
search_space=transbench101_micro
datasets=(jigsaw class_object class_scene)

# other variables:
trials=100
end_seed=$(($start_seed + $trials - 1))
test_size=100

# create config files
for dataset in ${datasets[@]}
do
 for i in $(seq 0 $((${#predictors[@]}-1)) )
 do
    predictor=${predictors[$i]}
    experiment_type=${experiment_types[$i]}
    python $base_file/create_configs.py --predictor $predictor --test_size $test_size \
	    --start_seed $start_seed --trials $trials --out_dir $out_dir --dataset=$dataset \
	    --search_space $search_space
 done
done

# run experiments
for t in $(seq $start_seed $end_seed)
do
 for dataset in ${datasets[@]}
 do
    for predictor in ${predictors[@]}
    do
        config_file=$out_dir/$dataset/configs/predictors/config\_$predictor\_$t.yaml
        echo ================running $predictor trial: $t =====================
        python naslib/runners/runner.py --config-file $config_file
    done
 done
done

