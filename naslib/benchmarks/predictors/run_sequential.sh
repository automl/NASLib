predictors=(jacov snip lcsvr \
valloss valacc sotl sotle \
gcn bonas seminas feedforward bananas \
gbdt xgb ngb rf \
dngo bohamiann bayes_lin_reg \
gp sparse_gp var_sparse_gp)

experiment_types=(single single vary_both \
vary_fidelity vary_fidelity vary_fidelity vary_fidelity \
vary_train_size vary_train_size vary_train_size vary_train_size vary_train_size \
vary_train_size vary_train_size vary_train_size vary_train_size \
vary_train_size vary_train_size vary_train_size \
vary_train_size vary_train_size vary_train_size)

# folders:
out_dir=p201_c10_jan9
base_file=NASLib/naslib
save_to_s3=true

# search space / data:
search_space=nasbench201
dataset=cifar10

# trials / seeds:
trials=100
start_seed=$1
end_seed=$(($start_seed + $trials - 1))
if [ -z "$start_seed" ]
then
    start_seed=0
fi

# dataset sizes:
test_size=200

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
        python $base_file/benchmarks/upload_to_s3.py --out_dir $out_dir
    fi
done
