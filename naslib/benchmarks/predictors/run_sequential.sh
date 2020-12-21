

predictors=(valloss valacc sotl bananas feedforward gbdt gcn bonas_gcn xgb \
ngb rf jacov dngo bohamiann lcnet bayes_lin_reg)
experiment_types=(vary_fidelity vary_fidelity vary_fidelity vary_train_size \
vary_train_size vary_train_size vary_train_size vary_train_size vary_train_size \
vary_train_size vary_train_size single vary_train_size vary_train_size \
vary_train_size vary_train_size)

out_dir=run
search_space=nasbench201
dataset=cifar10

start_seed=$1
trials=50

end_seed=$(($start_seed + $trials - 1))
test_size=200

# create config files
for i in $(seq 0 $((${#predictors[@]}-1)) )
do
    predictor=${predictors[$i]}
    experiment_type=${experiment_types[$i]}
    python naslib/utils/create_configs.py --predictor $predictor --experiment_type $experiment_type \
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
     python naslib/benchmarks/predictors/runner.py --config-file $config_file
 done
done
