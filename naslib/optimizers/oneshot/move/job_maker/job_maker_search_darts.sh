#!/bin/bash


seed="1 5 7 10 14 236"
epochs=100 #(warm_start_epochs+10)+(7*masking_interval)
data_size=50000
warm_start_epochs=(1 2 5)
instantenous="False"
dataset="cifar10"
masking_interval=(25 30 40)
search_space="darts"
training_portion=(50 10 95)
batch_size=64

base_path="/work/dlclarge2/agnihotr-ml/NASLib/naslib/optimizers/oneshot/movement/find_darts/search_prox_darts"


for s in $search_space
do
        for tp in ${training_portion[@]}
        do
                for m in ${masking_interval[@]}
                do
                        if [[ m -eq 1 ]]
                        then
                        #unset $instantenous
                        instantenous="True"                
                        else
                        #unset $instantenous
                        instantenous="False"
                        fi
                        for ds in $dataset
                        do
                                for wst in ${warm_start_epochs[@]}
                                do
                                        for sd in $seed
                                        do
                                                ((mask_epochs=m*7))
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
                                                out_dir="${base_path}/warm_${wst}_mask${m}_train_${train_p}"
                                                sbatch job_maker/jobs_darts_prox.sh $sd $epochs $wst $instantenous $ds $m $s $train_p $batch_size $data_size $out_dir

                                                #echo $out_dir
                                                #echo "seed" $sd "epochs" $epochs "warm_start_epochs" $wst "instantenous" $instantenous "dataset" $ds \
                                                #"masking interval" $m "search space" $s "training portion" $train_p "batch size" $batch_size "data size" $data_size
                                                #echo "warm_${wst}_mask${m}_train_${tp}"
                                                #break 5
                                                #echo $s $instantenous $batch_size $ds
                                        done
                                done
                        done
                done
        done
done
                        


