predictors=(fisher grad_norm grasp jacov snip synflow)

start_seed=$1
if [ -z "$start_seed" ]
then
    start_seed=0
fi

# folders:
base_file=scripts
s3_folder=p201_c10
out_dir=$s3_folder\_$start_seed

# search space / data:
search_space=nasbench201
dataset=cifar10

# other variables:
trials=1
end_seed=$(($start_seed + $trials - 1))
test_size=5

# create config files
for predictor in ${predictors[@]}
do
    python $base_file/create_configs.py --predictor $predictor \
    --test_size $test_size --start_seed $start_seed --trials $trials --out_dir $out_dir \
    --dataset=$dataset --search_space $search_space
done

# run experiments
#for t in $(seq $start_seed $end_seed)
#do
    #for predictor in ${predictors[@]}
    #do
        #config_file=$out_dir/$dataset/configs/predictors/config\_$predictor\_$t.yaml
        #echo ================running $predictor trial: $t =====================
        #python naslib/runners/runner.py --config-file $config_file
    #done
#done
