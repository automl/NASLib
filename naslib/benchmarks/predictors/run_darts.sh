predictors=(valloss valacc sotl bananas \
feedforward gbdt gcn bonas xgb \
ngb rf jacov dngo bohamiann \
bayes_lin_reg lcsvr gp sparse_gp \
var_sparse_gp seminas) # lcnet

experiment_types=(vary_fidelity vary_fidelity vary_fidelity vary_train_size \
vary_train_size vary_train_size vary_train_size vary_train_size vary_train_size \
vary_train_size vary_train_size single vary_train_size vary_train_size \
vary_train_size vary_train_size vary_fidelity vary_train_size \
vary_train_size vary_train_size) # vary_train_size

predictors=(sotl rf gp sparse_gp var_sparse_gp seminas)
experiment_types=(vary_fidelity vary_train_size vary_train_size vary_train_size vary_train_size vary_train_size)

predictors=(sotl rf gp sparse_gp var_sparse_gp ngb)
experiment_types=(vary_fidelity vary_train_size vary_train_size vary_train_size vary_train_size vary_train_size)

# folders:
out_dir=run
base_file=NASLib/naslib

# search space / data:
search_space=darts
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
    python $base_file/utils/create_configs.py --predictor $predictor --experiment_type $experiment_type \
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
done
