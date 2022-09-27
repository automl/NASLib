#!/bin/bash


#SBATCH -p alldlc_gpu-rtx2080 #testdlc_gpu-rtx2080 #mlhiwi_gpu-rtx2080
#SBATCH -J nb201_full_train
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/nb201_%A_%a.out
#SBATCH --error=slurm/nb201_%A_%a.err
#SBATCH -a [1-500]%10


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
training_portion=(10 50)
batch_size=128
optimizer="movement movement_test"
abs="False"

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
                                                	for opt in $optimizer
	                                                do
	                                                        ((mask_epochs=m*4))
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
	                                                        if [[ $opt == 'movement' ]]                                                  
	                                                        then
	                                                                abs="True"
		                                                        else
	                                                                abs="False"
	                                                        fi
	                                                        out_dir="${base_path}/warm_${wst}_mask_${m}_train_${train_p}_inst_${ac}_abs_${abs}"
	                                                        python runner.py --config-file config.yaml seed $sd search.epochs $epochs search.warm_start_epochs $wst search.instantenous $ac dataset $ds search.masking_interval $m search_space $s search.train_portion $train_p search.batch_size $batch_size search.data_size $data_size out_dir $out_dir optimizer $opt
	                                                        #echo $abs
	                                                        #((count=count+1))
	                                                        #echo "job_maker/jobs_darts_prox.sh $sd $epochs $wst $ac $ds $m $s $train_p $batch_size $data_size $out_dir"
	                                                done
                                                done
                                        done
                                done
                        done
                done
        done
done                        


