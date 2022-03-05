#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake #bosch_gpu-rtx2080    #bosch_cpu-cascadelake  # partition (queue)
#SBATCH -t 0-07:00           # time (D-HH:MM)
#SBATCH -o slurmlog/%A.%N.out       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e slurmlog/%A.%N.err       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J bbo-exps              # sets the job name. 
#SBATCH --mem=7G  

# Print some information about the job to STDOUT

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# python -u runner.py --config-file $1

for config_file_seed in $1/*
	do
		echo submitted ${config_file_seed}
		python -u runner.py --config-file $config_file_seed
	done
 
# echo $COMMAND;
# eval $COMMAND;

echo "DONE";
echo "Finished at $(date)";