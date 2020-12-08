
predictors=(sovl_50 sotl_50 bananas feedforward gbdt gcn)

out_dir=run
dataset=cifar10

start_seed=0
trials=200
end_seed=$(($start_seed + $trials - 1))
train_size=50
test_size=100

# create config files
for predictor in ${predictors[@]}
do
    python naslib/utils/create_configs.py --predictor $predictor --train_size $train_size \
    --test_size $test_size --start_seed $start_seed --trials $trials --out_dir $out_dir \
    --dataset=$dataset --config_type predictor
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

