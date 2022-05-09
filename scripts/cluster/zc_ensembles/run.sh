#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake #,ml_gpu-rtx2080 #ml_gpu-rtx2080     # bosch_gpu-rtx2080    #alldlc_gpu-rtx2080     # partition (queue)
#SBATCH -o logs/%x.%A-%a.%N.out       # STDOUT  %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%x.%A-%a.%N.err       # STDERR  %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -a 1 # array size
#SBATCH --job-name="ZC_ENSEMBLE"

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

searchspace=$1
dataset=$2
start_seed=$3
seed=$4

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

if [ -z "$start_seed" ]
then
    echo "Start seed not provided"
    exit 1
fi

if [ -z "$seed" ]
then
    echo "seed not provided"
    exit 1
fi

start=`date +%s`

python naslib/runners/bbo/runner.py --config-file configs/zc_ensemble/bananas/${searchspace}-${start_seed}/${dataset}/config_zc_${seed}.yaml

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
