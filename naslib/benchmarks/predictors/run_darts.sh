predictors=(fisher grad_norm grasp jacov snip synflow \
lce lce_m sotl sotle valacc valloss \
lcsvr omni_ngb omni_seminas \
bananas bonas gcn mlp nao seminas \
lgb ngb rf xgb \
bayes_lin_reg bohamiann dngo \
gp sparse_gp var_sparse_gp)

experiment_types=(single single single single single single \
vary_fidelity vary_fidelity vary_fidelity vary_fidelity vary_fidelity vary_fidelity \
vary_both vary_both vary_both \
vary_train_size vary_train_size vary_train_size vary_train_size vary_train_size vary_train_size \
vary_train_size vary_train_size vary_train_size vary_train_size \
vary_train_size vary_train_size vary_train_size \
vary_train_size vary_train_size vary_train_size)

start_seed=$1
if [ -z "$start_seed" ]
then
    start_seed=0
fi

# folders:
base_file=NASLib/naslib
s3_folder=p301
out_dir=$s3_folder\_$start_seed

# search space / data:
search_space=darts
dataset=cifar10

# other variables:
trials=100
end_seed=$(($start_seed + $trials - 1))
save_to_s3=true
test_size=100

# create config files
for i in $(seq 0 $((${#predictors[@]}-1)) )
do
    predictor=${predictors[$i]}
    experiment_type=${experiment_types[$i]}
    python $base_file/benchmarks/create_configs.py --predictor $predictor --experiment_type $experiment_type \
    --test_size $test_size --start_seed $start_seed --trials $trials --out_dir $out_dir \
    --dataset=$dataset --config_type predictor --search_space $search_space
done

# run experiments
for t in $(seq $start_seed $end_seed)
do
    for predictor in ${predictors[@]}
    do
        config_file=$out_dir/$dataset/configs/predictors/config\_$predictor\_$t.yaml
        echo ================running $predictor trial: $t =====================
        python $base_file/benchmarks/predictors/runner.py --config-file $config_file
    done
    if [ "$save_to_s3" ]
    then
        # zip and save to s3
        echo zipping and saving to s3
        zip -r $out_dir.zip $out_dir 
        python $base_file/benchmarks/upload_to_s3.py --out_dir $out_dir --s3_folder $s3_folder
    fi
done
