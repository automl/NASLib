#!/bin/bash
#SBATCH -p meta_gpu-ti # partition (queue)
#SBATCH --mem 30000 # memory pool for all cores (8GB)
#SBATCH -t 11-00:00 # time (D-HH:MM)
#SBATCH -c 2 # number of cores
#SBATCH -a 1-8 # array size
#SBATCH --gres=gpu:1  # reserves four GPUs
#SBATCH -o logs/%A-%a.o
#SBATCH -e logs/%A-%a.e
#SBATCH -J NASBENCH-201 # sets the job name. If not specified, the file name will be used as job name
# Print some information about the job to STDOUT
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

# Activate conda environment
source ~/.bashrc
conda activate pytorch1.5

gpu_counter=1

for optimizer in "GDASOptimizer"; do
  for seed in {0..3}; do
    # Job to perform
    if [ $gpu_counter -eq $SLURM_ARRAY_TASK_ID ]; then
      PYTHONPATH=../../../. python runner_sdarts.py --seed=${seed} --optimizer=${optimizer}
      exit $?
    fi
    let gpu_counter+=1
  done
done
# Print some Information about the end-time to STDOUT
echo "DONE"
echo "Finished at $(date)"
