#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake #bosch_gpu-rtx2080    #bosch_cpu-cascadelake  # partition (queue)
#SBATCH -t 1-00:00           # time (D-HH:MM)
#SBATCH --output=/path/to/log/%A_%a.%N.out
#SBATCH --job-name=naslib-exps
#SBATCH --chdir=/path/to/your/workdir
#SBATCH --mem=40G
#SBATCH --array=1-12000

# array size needs to be set to amount of config files

# Print some information about the job to STDOUT
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";
/path/to/run_config.sh $SLURM_ARRAY_TASK_ID
 
echo "DONE";
echo "Finished at $(date)";
