#!/bin/bash
#SBATCH -p ml_gpu-rtx2080 #ml_gpu-rtx2080     # bosch_gpu-rtx2080    #alldlc_gpu-rtx2080     # partition (queue)
#SBATCH --gres=gpu:1          # reserves one GPU
#SBATCH -o logs-short/%x.%A-%a.%N.out       # STDOUT  %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs-short/%x.%A-%a.%N.err       # STDERR  %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J gdas-short        # search space - algorithm
#SBATCH -a 1-4 # array size

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

source /home/zelaa/NASLib/nl-venv/bin/activate

python runner.py --config-file config_short_eval.yaml --optimizer gdas --seed $SLURM_ARRAY_TASK_ID --eval-only

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
