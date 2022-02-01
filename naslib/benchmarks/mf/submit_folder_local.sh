#!/bin/bash
#SBATCH -o slurmlog/%A.%N.out       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e slurmlog/%A.%N.err       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J bbo-exps              # sets the job name. 
#SBATCH --mem=7G  

# Print some information about the job to STDOUT

echo "Workingdir: $PWD";
echo "Started at $(date)";

# python -u runner.py --config-file $1

for config_file_seed in $1/*
	do
		echo submitted ${config_file_seed}
		/Users/lars/Projects/naslib-venv/bin/python runner.py --config-file $config_file_seed
	done

echo "DONE";
echo "Finished at $(date)";