#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake #,ml_gpu-rtx2080 #ml_gpu-rtx2080     # bosch_gpu-rtx2080    #alldlc_gpu-rtx2080     # partition (queue)
#SBATCH -o logs/%x.%A-%a.%N.out       # STDOUT  %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%x.%A-%a.%N.err       # STDERR  %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -a 0-399:10 # array size
#SBATCH --job-name="THE_JOB_NAME"
#SBATCH --mem=5G
echo "Workingdir: $PWD";
echo "Started at $(date)";
#echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

searchspace=$1
dataset=$2
start_seed=$3
n_seeds=10
experiment=$5
optimizer=bananas
zc_usage=$6
zc_source=$7

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

if [ -z "$n_seeds" ]
then
    echo "n_seeds not provided"
    exit 1
fi

if [ -z "$experiment" ]
then
    echo "experiment not provided"
    exit 1
fi

if [ -z "$zc_usage" ]
then
    echo "zc_usage not provided"
    exit 1
fi

if [ -z "$zc_source" ]
then
    echo "zc_source not provided"
    exit 1
fi

start=`date +%s`

for i in $(seq 0 $(($n_seeds - 1)))
do
    echo "running experiment for config_$(($start_seed + $i)).yaml"
    python naslib/runners/bbo/runner.py --config-file naslib/configs/${experiment}/${zc_usage}/${zc_source}/${optimizer}/${searchspace}-${start_seed}/${dataset}/config_$(($start_seed + $i)).yaml
done

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
