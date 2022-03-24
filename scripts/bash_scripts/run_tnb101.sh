predictors=(fisher flops grad_norm grasp jacov params snip synflow)
search_space=transbench101_micro
datasets=(jigsaw class_object class_scene)
trials=1
s3_folder=tnb101
out_dir=$s3_folder
test_size=100
start_seed=1000
mkdir $out_dir
for j in $(seq 0 $((${#datasets[@]}-1)) )
do
   for i in $(seq 0 $((${#predictors[@]}-1)) )
   do
     predictor=${predictors[$i]}
     experiment_type=single
     dataset=${datasets[$j]}
     python scripts/create_configs.py --predictor $predictor --experiment_type $experiment_type \
     --test_size $test_size --start_seed $start_seed --trials $trials --out_dir $out_dir \
     --dataset=$dataset --config_type predictor --search_space $search_space
    done
done
# run experiments
t=1000
for dataset in ${datasets[@]}
do
    for predictor in ${predictors[@]}
    do
         config_file=$out_dir/$dataset/configs/predictors/config\_$predictor\_$t.yaml
         echo ================running $predictor trial: $t =====================
         python naslib/runners/runner.py --config-file $config_file
    done
done
