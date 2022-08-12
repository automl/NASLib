#!/bin/bash
#SBATCH -q dlc-moradias
#SBATCH -p mlhiwidlc_gpu-rtx2080 # partition (queue)
#SBATCH --gres=gpu:1
#SBATCH -o logs/%x.%N.%j.out # STDOUT
#SBATCH -e logs/%x.%N.%j.err # STDERR
#SBATCH -a 1-5 # array size
#SBATCH -J GDAS_101_EVAL # sets the job name. If not
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CUPS_PER_NODE gpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

source ~/.bashrc
conda activate edge_popup_fix

python naslib/runners/nas/runner.py --seed $SLURM_ARRAY_TASK_ID

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
