#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake #,ml_gpu-rtx2080 #ml_gpu-rtx2080     # bosch_gpu-rtx2080    #alldlc_gpu-rtx2080     # partition (queue)
#SBATCH -o logs/%x.memMEM_FOR_JOB.%A-%a.%N.out       # STDOUT  %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%x.memMEM_FOR_JOB.%A-%a.%N.err       # STDERR  %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -a JOB_ARRAY_RANGE # array size
#SBATCH --mem=MEM_FOR_JOB
#SBATCH --job-name="THE_JOB_NAME"

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

searchspace=$1
dataset=$2
predictor=$3
start_seed=$4
experiment=$5
N_MODELS=2500

SCRIPT_DIR="/home/jhaa/NASLib/scripts/vscode_remote_debugging"
while read var value
do
    export "$var"="$value"
done < $SCRIPT_DIR/config.conf

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

if [ -z "$predictor" ]
then
    echo "Predictor argument not provided"
    exit 1
fi

if [ -z "$start_seed" ]
then
    echo "Start seed not provided"
    exit 1
fi

if [ -z "$experiment" ]
then
    echo "experiment not provided"
    exit 1
fi

start=`date +%s`
python naslib/runners/zc/benchmarks/runner.py --config-file naslib/configs/${experiment}/${predictor}/${searchspace}-${start_seed}/${dataset}/config_${start_seed}.yaml start_idx ${SLURM_ARRAY_TASK_ID} n_models $N_MODELS

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
