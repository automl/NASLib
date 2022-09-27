#!/bin/bash
count=1
job_name=10000
#seed="0 4 10 15 29 38 76 224 45 1 2 3 5 7"
seed="0 4 10 77 69 79 58 55 189 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27"
#seed="28 29 30 31 32 34 35 36 37 39 40 41 42 43 44 46 47 48 49 50"
#seed="51 52 53 54 56 57 58 59 60 61 62 63 64 65 66 67 68 70 71 72 73 74 75"
#seed="90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115"
#seed="116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145"
epochs=100 #(warm_start_epochs+10)+(7*masking_interval)
data_size=50000
warm_start_epochs=(2)
instantenous="True"
dataset="cifar10"
masking_interval=(10)
search_space="darts"
training_portion=(10)
batch_size=64

base_path="/work/dlclarge2/agnihotr-ml/nas301_test_acc/NASLib/naslib/optimizers/oneshot/move/runs/abs/find_darts_10"


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
                                                        out_dir="${base_path}/warm_${wst}_mask_${m}_train_${train_p}_inst_${ac}"
                                                        sbatch job_maker/jobs_darts_prox.sh $sd $epochs $wst $ac $ds $m $s $train_p $batch_size $data_size $out_dir 
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


