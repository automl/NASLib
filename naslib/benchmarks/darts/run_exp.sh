#!/bin/bash
#SBATCH -q dlc-krishnan
#SBATCH -p mlhiwidlc_gpu-rtx2080
#SBATCH --gres=gpu:1                  # reserves one GPU
#SBATCH -o logs/%x.%A-%a.%N.out       # STDOUT  %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%x.%A-%a.%N.err       # STDERR  %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -a 0-7                        # array size
#SBATCH -J OneShot

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";
echo "Optimizer is $1"
start=`date +%s`

# Activate virtual env so that run_experiment can load the correct packages
# source /home/krishnan/miniconda3/bin/activate
# conda activate naslib

python runner_all.py --config-file /home/krishnan/naslib/NASLib/naslib/defaults/darts_defaults.yaml opt -f seed $SLURM_ARRAY_TASK_ID optimizer $1 search-space $2 dataset $3

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
