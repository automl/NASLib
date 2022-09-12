#!/bin/bash
count=1
job_name=10000
seed="0 4 10 15 29 38 76"
epochs=100 #(warm_start_epochs+10)+(7*masking_interval)
data_size=50000
warm_start_epochs=(2 5)
instantenous="True False"
dataset="cifar10"
masking_interval=(5 10 15 20)
search_space="nasbench201"
training_portion=(95)
batch_size=128

base_path="/work/dlclarge2/agnihotr-ml/nas301_test_acc/NASLib/naslib/optimizers/oneshot/move/runs/abs/find_nb201"


for s in $search_space
do
        for tp in ${training_portion[@]}
        do
                for m in ${masking_interval[@]}
                do
                        for ac in $instantenous
                        do
                                for ds in $dataset
                                do
                                        for wst in ${warm_start_epochs[@]}
                                        do
                                                for sd in $seed
                                                do
                                                        ((mask_epochs=m*5))
                                                        ((data_size=tp*500))
                                                        ((epochs=wst+mask_epochs))
                                                        if [[ tp -eq 10 ]]
                                                        then
                                                                train_p="0.1"
                                                        elif [[ tp -eq 50 ]]
                                                        then
                                                                train_p="0.5"
                                                        else
                                                                train_p="0.95"
                                                        fi
                                                        out_dir="${base_path}/warm_${wst}_mask_${m}_train_${train_p}_inst_${ac}"
                                                        sbatch -J nb201-c10 job_maker/jobs_darts_prox.sh $sd $epochs $wst $ac $ds $m $s $train_p $batch_size $data_size $out_dir $job_name
                                                        #echo $count
                                                        #((count=count+1))
                                                        #echo "job_maker/jobs_darts_prox.sh $sd $epochs $wst $ac $ds $m $s $train_p $batch_size $data_size $out_dir"
                                                done
                                        done
                                done
                        done
                done
        done
done                        


