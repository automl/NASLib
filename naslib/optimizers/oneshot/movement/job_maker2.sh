#!/bin/bash


seed=0
epochs=100 #(warm_start_epochs+10)+(7*masking_interval)
warm_start_epochs=(1 3 5 10 20)
instantenous="False"
dataset="cifar10 cifar100 ImageNet16-120"
masking_interval=(1 3 5 10)
search_space="nasbench201 darts"
training_portion="0.1 0.5 0.95"
batch_size=128

for s in $search_space
do
        if [[ $s = "darts" ]]
        then
            #unset $batch_size
            #unset $dataset
            ((batch_size=32))
            dataset="cifar10"
        else
            dataset="cifar10 cifar100 ImageNet16-120"
            ((batch_size=128))
        fi
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
                for wst in ${warm_start_epochs[@]}
                do
                        for ds in $dataset
                        do
                                for tp in $training_portion
                                do
                                        ((seed=seed+1))
                                        ((mask_epochs=m*7))
                                        ((epochs=wst+mask_epochs+5))
                                        sbatch jobs1.sh $seed $epochs $wst $instantenous $ds $m $s $tp $batch_size
                                        
                                        echo "seed" $seed "epochs" $epochs "warm_start_epochs" $wst "instantenous" $instantenous "dataset" $ds \
                                        "masking interval" $m "search space" $s "training portion" $tp "batch size" $batch_size
                                        #break 5
                                        #echo $s $instantenous $batch_size $ds
                                done
                        done
                done
        done
done
                        


