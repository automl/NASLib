#!/bin/bash

#SBATCH -p alldlc_gpu-rtx2080 #testdlc_gpu-rtx2080 #mlhiwi_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/looping/looper_10_%A.out
#SBATCH --error=slurm/looping/looper_10_%A.err



echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

python runner_loop.py --config-file config.yaml 

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
