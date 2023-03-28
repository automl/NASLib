#!/bin/bash
#SBATCH -p testdlc_gpu-rtx2080#alldlc_gpu-rtx2080 #bosch_gpu-rtx2080 #bosch_cpu-cascadelake #bosch_gpu-rtx2080 #mldlc_gpu-rtx2080 #alldlc_gpu-rtx2080 #gpu_tesla-P100     #ml_gpu-rtx2080     # bosch_gpu-rtx2080    #alldlc_gpu-rtx2080     # partition (queue)
#SBATCH -t 0-02:00           # time (D-HH:MM)
#SBATCH -o slurmlog/%A.%N.out       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e slurmlog/%A.%N.err       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J bbo-exps              # sets the job name. 
#SBATCH --mem=10G  

# Print some information about the job to STDOUT

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

user="robertsj"
runner_dir="/home/$user/NASLib/naslib/runners/bbo"

# source $CONDA_SOURCE
# conda activate $CONDA_ENV

# python -u $runner_dir/runner.py --config-file $1

for config_file_seed in $1/*
    do
        echo submitted ${config_file_seed}
        python -u $runner_dir/runner.py --config-file $config_file_seed
        # python -u -m debugpy --listen 0.0.0.0:$PORT --wait-for-client $runner_dir/runner.py --config-file $config_file_seed
    done
 
# echo $COMMAND;
# eval $COMMAND;

# conda deactivate
echo "DONE";
echo "Finished at $(date)"; 
