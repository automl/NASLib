#!/bin/bash

#SBATCH -p alldlc_gpu-rtx2080 #testdlc_gpu-rtx2080 #mlhiwi_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/workshop_looping/gsparsity_95_%A.out
#SBATCH --error=slurm/workshop_looping/gsparsity_95_%A.err



echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

python runner_loop.py --config-file configs/config_gsparsity.yaml 

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
