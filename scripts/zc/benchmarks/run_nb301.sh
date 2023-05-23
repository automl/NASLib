#!/bin/bash

experiment=$1
predictor=$2
start_seed=9000

if [ -z "$experiment" ]
then
    echo "Experiment argument not provided"
    exit 1
fi

if [ -z "$predictor" ];
then
    predictors=(zen) #fisher  grad_norm grasp jacov snip synflow epe_nas flops params plain l2_norm nwot)
    memory=(    32G)  #64G     32G       64G   32G   32G  32G     32G     5G    5G     32G   32G     32G)

    # predictors=(nwot)
    # memory=(64G)
else
    predictors=($predictor)
fi

searchspace=nasbench301
datasets=(cifar10)

start=0
end=11221
n_models=2500
range=${start}-${end}:${n_models}

sed -i "s/JOB_ARRAY_RANGE/$range/" ./scripts/zc/benchmarks/run.sh
sed -i "s/JOB_N_MODELS/$n_models/" ./scripts/zc/benchmarks/run.sh

for dataset in "${datasets[@]}"
do
    for pred_index in "${!predictors[@]}"
    do
        pred="${predictors[$pred_index]}"
        mem="${memory[$pred_index]}"

        sed -i "s/MEM_FOR_JOB/$mem/" ./scripts/zc/benchmarks/run.sh
        sed -i "s/THE_JOB_NAME/${searchspace}-${dataset}-${pred}/" ./scripts/zc/benchmarks/run.sh

        echo $pred $dataset
        sbatch ./scripts/zc/benchmarks/run.sh $searchspace $dataset $pred $start_seed $experiment --bosch
        # cat ./scripts/zc/benchmarks/run.sh # $searchspace $dataset $pred $start_seed $experiment --bosch

        sed -i "s/${searchspace}-${dataset}-${pred}/THE_JOB_NAME/" ./scripts/zc/benchmarks/run.sh
        sed -i "s/#SBATCH --mem=${mem}/#SBATCH --mem=MEM_FOR_JOB/" ./scripts/zc/benchmarks/run.sh

        sed -i "s/x.mem${mem}.%A-%a.%N.out/x.memMEM_FOR_JOB.%A-%a.%N.out/" ./scripts/zc/benchmarks/run.sh
        sed -i "s/x.mem${mem}.%A-%a.%N.err/x.memMEM_FOR_JOB.%A-%a.%N.err/" ./scripts/zc/benchmarks/run.sh
    done

    echo ""
done

# Restore placeholders
sed -i "s/#SBATCH -a $range/#SBATCH -a JOB_ARRAY_RANGE/" ./scripts/zc/benchmarks/run.sh
sed -i "s/N_MODELS=$n_models/N_MODELS=JOB_N_MODELS/" ./scripts/zc/benchmarks/run.sh
