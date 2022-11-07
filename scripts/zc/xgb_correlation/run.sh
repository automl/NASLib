#!/bin/bash
#SBATCH -p testdlc_gpu-rtx2080 #bosch_cpu-cascadelake #,ml_gpu-rtx2080 #ml_gpu-rtx2080     # bosch_gpu-rtx2080    #alldlc_gpu-rtx2080     # partition (queue)
#SBATCH -o logs/%x.%A-%a.%N.out       # STDOUT  %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%x.%A-%a.%N.err       # STDERR  %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -a 0 # array size
#SBATCH --mem=5G
#SBATCH --job-name="XGB_ZC_CORRELATION"

searchspace=$1
dataset=$2
train_size=train_size_$3
start_seed=$4
experiment=$5
k=k_$6
n_seeds=100

if [ -z "$searchspace" ]
then
    echo "Search space argument not provided"
    exit 1
fi

if [ -z "$dataset" ]
then
    echo "Dataset argument not provided"
    exit 1
fi

if [ -z "$train_size" ]
then
    echo "Train size argument not provided"
    exit 1
fi

if [ -z "$start_seed" ]
then
    echo "Start seed not provided"
    exit 1
fi

if [ -z "$experiment" ]
then
    echo "Experiment argument not provided"
    exit 1
fi

start=`date +%s`
for t in $(seq 0 $n_seeds)
do
    seed=$(($start_seed + $t))
    python naslib/runners/zc/bbo/xgb_runner.py --config-file naslib/configs/${experiment}/${train_size}/$k/${searchspace}-${start_seed}/${dataset}/config_${seed}.yaml
done

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
