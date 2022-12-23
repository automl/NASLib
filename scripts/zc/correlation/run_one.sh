#!/bin/bash
#SBATCH -p testdlc_gpu-rtx2080 #bosch_cpu-cascadelake #,ml_gpu-rtx2080 #ml_gpu-rtx2080     # bosch_gpu-rtx2080    #alldlc_gpu-rtx2080     # partition (queue)
#SBATCH -o logs/%x.%A-%a.%N.out       # STDOUT  %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%x.%A-%a.%N.err       # STDERR  %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -a 0 # array size
#SBATCH --mem=16GB
#SBATCH --job-name="ZC_ONE_RUN"

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

searchspace=$1
predictor=$3
dataset=$2
seed=$4
experiment=correlation

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

if [ -z "$seed" ]
then
    echo "Seed not provided"
    exit 1
fi

if [ -z "$experiment" ]
then
    echo "experiment not provided"
    exit 1
fi

start=`date +%s`

python naslib/runners/zc/runner.py --config-file configs/${experiment}/${predictor}/${searchspace}-9000/${dataset}/config_${seed}.yaml

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
