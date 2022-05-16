#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake #,ml_gpu-rtx2080 #ml_gpu-rtx2080     # bosch_gpu-rtx2080    #alldlc_gpu-rtx2080     # partition (queue)
#SBATCH -o logs/%x.%A-%a.%N.out       # STDOUT  %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%x.%A-%a.%N.err       # STDERR  %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -a 0-7 # array size
#SBATCH --mem=16G
#SBATCH --job-name="CREATE_BENCHMARK"

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

searchspace=$1
dataset=$2
train_size=train_size_$3
start_seed=$4
experiment=$5

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

seed=$(($start_seed + ${SLURM_ARRAY_TASK_ID}))
python naslib/runners/bbo/xgb_runner.py --config-file configs/${experiment}/${train_size}/${searchspace}-${start_seed}/${dataset}/config_${seed}.yaml

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
