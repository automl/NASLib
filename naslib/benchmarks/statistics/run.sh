search_spaces=(nasbench101 nasbench201 nasbench201 nasbench201 \
darts nlp transbench101 transbench101 transbench101 transbench101 \
transbench101 transbench101 transbench101)

datasets=(cifar10 cifar10 cifar100 ImageNet16-120 \
cifar10 penntreebank class_scene class_object jigsaw room_layout \
segmentsemantic normal autoencoder)

run_acc_stats=(1 1 1 1 \
1 1 1 1 1 1 \
1 1 1)

run_nbhd_sizes=(1 1 1 1 \
1 1 1 1 1 1 \
1 1 1)

run_autocorrs=(1 1 1 1 \
1 1 1 1 1 1 \
1 1 1)

start_seed=$1
if [ -z "$start_seed" ]
then
    start_seed=0
fi

# folders:
base_file=NASLib/naslib
s3_folder=stats
out_dir=$s3_folder\_$start_seed

# other variables:
trials=3
end_seed=$(($start_seed + $trials - 1))
save_to_s3=true

max_set_size=430000
max_nbhd_trials=500
max_autocorr_trials=20
walks=5000

# create config files
for i in $(seq 0 $((${#search_spaces[@]}-1)) )
do
    search_space=${search_spaces[$i]}
    dataset=${datasets[$i]}
    run_acc_stats=${run_acc_stats[$i]}
    run_nbhd_size=${run_nbhd_sizes[$i]}
    run_autocorr=${run_autocorrs[$i]}

    python $base_file/benchmarks/create_configs.py --search_space $search_space --dataset=$dataset \
    --run_acc_stats $run_acc_stats --run_nbhd_size $run_nbhd_size --run_autocorr $run_autocorr \
    --start_seed $start_seed --trials $trials --out_dir $out_dir --max_set_size $max_set_size \
    --max_nbhd_trials $max_nbhd_trials --max_autocorr_trials $max_autocorr_trials --walks $walks \
    --config_type statistics
done

# run experiments
for t in $(seq $start_seed $end_seed)
do
    for i in $(seq 0 $((${#search_spaces[@]}-1)) )
    do
        search_space=${search_spaces[$i]}
        dataset=${datasets[$i]}
        config_file=$out_dir/$search_space/$dataset/configs/statistics/config\_$t.yaml
        echo ================running $search_space $dataset trial: $t =====================
        python $base_file/benchmarks/statistics/runner.py --config-file $config_file
    done
    if [ "$save_to_s3" ]
    then
        # zip and save to s3
        echo zipping and saving to s3
        zip -r $out_dir.zip $out_dir 
        python $base_file/benchmarks/upload_to_s3.py --out_dir $out_dir --s3_folder $s3_folder
    fi
done
