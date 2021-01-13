
#optimizers=(rs re ls bp bananas)
optimizers=(rs re ls)
epochs=150
out_dir=run
dataset=cifar10

start_seed=0
trials=200
end_seed=$(($start_seed + $trials - 1))

# create config files
for opt in ${optimizers[@]}
do
    python naslib/utils/create_configs.py --optimizer $opt --epochs $epochs \
    --start_seed $start_seed --trials $trials --out_dir $out_dir --dataset=$dataset
done

# run experiments
for t in $(seq $start_seed $end_seed)
do
    for opt in ${optimizers[@]}
    do
        config_file=$out_dir/$dataset/configs/nas/config\_$opt\_$t.yaml
        echo ================running $opt trial: $t epochs: $epochs=====================
        python naslib/benchmarks/nasbench201/runner.py --config-file $config_file
    done
done

# remove config files
