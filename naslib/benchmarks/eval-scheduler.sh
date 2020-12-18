#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080 #gpu_tesla-P100     #ml_gpu-rtx2080     # bosch_gpu-rtx2080    #alldlc_gpu-rtx2080     # partition (queue)
#SBATCH --mem 4000            # memory pool for all cores (4GB)
#SBATCH -t 10-00:00           # time (D-HH:MM)
#SBATCH -c 1                  # number of cores
#SBATCH --gres=gpu:1          # reserves one GPU
#SBATCH -o %x.%A.%N.out       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e %x.%A.%N.err       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH --mail-type=END,FAIL  # (recive mails about end and timeouts/crashes of your job)
#SBATCH -J hiera              # sets the job name. 

# Print some information about the job to STDOUT
#set -x

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Activate virtual env so that run_experiment can load the correct packages
source /home/ruchtem/dev/venvs/naslib/bin/activate


if [ -z "$3" ]; then
    COMMAND="python -u runner.py --config-file $1 --eval-only optimizer $2";
else
    COMMAND="python -u runner.py --config-file $1 --eval-only optimizer $2 seed $3";
fi

echo $COMMAND;
eval $COMMAND;

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";