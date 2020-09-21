#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080 #gpu_tesla-P100     #ml_gpu-rtx2080     # bosch_gpu-rtx2080    #alldlc_gpu-rtx2080     # partition (queue)
#SBATCH --mem 4000            # memory pool for all cores (4GB)
#SBATCH -t 0-10:00            # time (D-HH:MM)
#SBATCH -c 1                  # number of cores
#SBATCH --gres=gpu:1          # reserves one GPU
#SBATCH -o %x.%A.%N.out       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e %x.%A.%N.err       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH --mail-type=END,FAIL  # (recive mails about end and timeouts/crashes of your job)
#SBATCH -J nb2-darts          # search space - algorithm

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

###################
# commands here

# Activate virtual env so that run_experiment can load the correct packages
source /home/ruchtem/dev/venvs/naslib/bin/activate

# currently only cifar-10
python runner.py --config-file config.yaml --seed 1 --optimizer darts


#
###################

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
