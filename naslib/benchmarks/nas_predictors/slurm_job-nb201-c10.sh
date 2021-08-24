#!/bin/bash
#SBATCH -p ml_gpu-teslaP100 #ml_gpu-rtx2080     # bosch_gpu-rtx2080    #alldlc_gpu-rtx2080     # partition (queue)
#SBATCH --gres=gpu:1          # reserves one GPU
#SBATCH -o logs_bo-201-c10/%x.%A-%a.%N.out       # STDOUT  %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs_bo-201-c10/%x.%A-%a.%N.err       # STDERR  %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -D .
#SBATCH -a 17-23 # array size

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

# Activate virtual env so that run_experiment can load the correct packages
source activate python37
python runner.py --config-file bo201_feb22_0/cifar10/configs/nas_predictors/config_bananas_${1}_${SLURM_ARRAY_TASK_ID}.yaml
#python runner.py --config-file bo201_c100_feb01_0/cifar100/configs/nas_predictors/config_bananas_${1}_${SLURM_ARRAY_TASK_ID}.yaml
#python runner.py --config-file bo201_imagenet_feb01_0/ImageNet16-120/configs/nas_predictors/config_bananas_${1}_${SLURM_ARRAY_TASK_ID}.yaml


end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
