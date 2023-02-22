#!/bin/bash

#SBATCH -p mlhiwidlc_gpu-rtx2080 #testdlc_gpu-rtx2080 #mlhiwi_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/testing/mean_drop_%A.out
#SBATCH --error=slurm/testing/mean_drop_%A.err



echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

python runner_loop.py --config-file configs/mean_drop_one_config2.yaml

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
