#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080 #gpu_tesla-P100     #ml_gpu-rtx2080     # bosch_gpu-rtx2080    #alldlc_gpu-rtx2080     # partition (queue)
#SBATCH --mem 4000            # memory pool for all cores (4GB)
#SBATCH -t 0-02:00            # time (D-HH:MM)
#SBATCH -c 1                  # number of cores
#SBATCH --gres=gpu:1          # reserves one GPU
#SBATCH -o %x.%A.%N.out       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e %x.%A.%N.err       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J test        # sets the job name.

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";


start=`date +%s`

###################
# commands here



# Activate virtual env so that run_experiment can load the correct packages
source /home/ruchtem/dev/venvs/naslib/bin/activate

python test.py --optimizer gdas



#
###################

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
