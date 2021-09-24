search_spaces=(nasbench101 nasbench201 nasbench201 nasbench201 \
asr darts nlp transbench101_micro transbench101_micro \
transbench101_micro transbench101_micro transbench101_micro \
transbench101_micro transbench101_micro transbench101_macro \
transbench101_macro transbench101_macro transbench101_macro \
transbench101_macro transbench101_macro transbench101_macro)

datasets=(cifar10 cifar10 cifar100 ImageNet16-120 \
timit cifar10 penntreebank class_scene class_object \
jigsaw room_layout segmentsemantic \
normal autoencoder class_scene \
class_object jigsaw room_layout \
segmentsemantic normal autoencoder)

predictors=(nao gp xgb rf bohamiann)

start_hposeed=$1
if [ -z "$start_hposeed" ]
then
    start_hposeed=0
fi

# folders:
base_file=NASLib/naslib
save_to_s3=true
s3_folder=predictor_hpo_sep24
out_dir=$s3_folder\_$start_hposeed
hpo_config_folder=predictor_hpo

# there are two types of seeds:
# 'hposeed' for the random hyperparameters
# 'seed' for the randomness within the algorithm 

hpo_trials=1000
end_hposeed=$(($start_hposeed + $hpo_trials - 1))
start_seed=0
end_seed=20

train_size=100
test_size=200

for hposeed in $(seq $start_hposeed $end_hposeed)
do
    for i in $(seq 0 $((${#search_spaces[@]}-1)) )
    do
        search_space=${search_spaces[$i]}
        dataset=${datasets[$i]}
        for predictor in ${predictors[@]}
        do
            for seed in $(seq $start_seed $end_seed)
            do
                # create experiment configs
                base_folder=$out_dir/hpo\_$hposeed/$search_space/$dataset
                save_folder=$base_folder/$predictor/$seed
                config_file=$base_folder/configs/$predictor\_$seed.yaml
                python $base_file/benchmarks/generate_predictor_hpo_configs.py --search_space \
                $search_space --dataset $dataset --predictor $predictor --hposeed $hposeed \
                --seed $seed --train_size_single $train_size --test_size $test_size --out_dir $out_dir \
                --save_folder $save_folder --config_file $config_file --hpo_config_folder \
                $hpo_config_folder

                # run the experiment
                echo =========running hpo $hposeed $search_space $dataset $predictor trial: $seed =============
                python $base_file/benchmarks/predictors/runner.py --config-file $config_file
            done
        done
    done
    #if [ "$save_to_s3" ]
    #then
        # zip and save to s3
        #echo zipping and saving to s3
        #zip -r $out_dir.zip $out_dir 
        #python $base_file/benchmarks/upload_to_s3.py --out_dir $out_dir --s3_folder $s3_folder
    #fi
done