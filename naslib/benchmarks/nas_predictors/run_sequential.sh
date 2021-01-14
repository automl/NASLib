optimizer=bananas
predictors=(bananas feedforward gbdt gcn bonas xgb ngb rf dngo \
bohamiann bayes_lin_reg seminas gp sparse_gp var_sparse_gp)


# folders:
out_dir=run
base_file=NASLib/naslib

# search space / data:
search_space=nasbench201
dataset=cifar10
search_epochs=500

# trials / seeds:
trials=100
start_seed=$1
end_seed=$(($start_seed + $trials - 1))
if [ -z "$start_seed" ]
then
    start_seed=0
fi

# create config files
for i in $(seq 0 $((${#predictors[@]}-1)) )
do
    predictor=${predictors[$i]}
    python $base_file/utils/create_configs.py --predictor $predictor \
    --epochs $search_epochs --start_seed $start_seed --trials $trials \
    --out_dir $out_dir --dataset=$dataset --config_type nas_predictor \
    --search_space $search_space --optimizer $optimizer
done

# run experiments
for t in $(seq $start_seed $end_seed)
do
 for predictor in ${predictors[@]}
 do
     config_file=$out_dir/$dataset/configs/nas_predictors/config\_$optimizer\_$predictor\_$t.yaml
     echo ================running $predictor trial: $t =====================
     python $base_file/benchmarks/nas_predictors/runner.py --config-file $config_file
 done
done
