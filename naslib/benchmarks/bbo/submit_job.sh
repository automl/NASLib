#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080 #gpu_tesla-P100     #ml_gpu-rtx2080     # bosch_gpu-rtx2080    #alldlc_gpu-rtx2080     # partition (queue)
#SBATCH -t 0-01:00           # time (D-HH:MM)
#SBATCH -c 8                  # number of cores
#SBATCH -o slurmlog/%A.%N.out       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e slurmlog/%A.%N.err       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J bbo-exps              # sets the job name. 

# Print some information about the job to STDOUT

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# python -u naslib/benchmarks/bbo/runner.py --config-file $1
python -u runner.py --config-file $1

# echo $COMMAND;
# eval $COMMAND;

echo "DONE";
echo "Finished at $(date)";