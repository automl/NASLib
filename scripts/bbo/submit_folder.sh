#!/bin/bash
#SBATCH -p testdlc_gpu-rtx2080 #alldlc_gpu-rtx2080 #bosch_gpu-rtx2080 #bosch_cpu-cascadelake #bosch_gpu-rtx2080 #mldlc_gpu-rtx2080 #alldlc_gpu-rtx2080 #gpu_tesla-P100     #ml_gpu-rtx2080     # bosch_gpu-rtx2080    #alldlc_gpu-rtx2080     # partition (queue)
#SBATCH -t 01:00:00           # time (D-HH:MM:SS)
#SBATCH -o slurmlog/%A.%N.out       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e slurmlog/%A.%N.err       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J bbo-exps              # sets the job name. 
#SBATCH --mem=10G  

# Print some information about the job to STDOUT

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SCRIPT_DIR="/home/jhaa/NASLib/scripts/vscode_remote_debugging"
while read var value
do
    export "$var"="$value"
done < $SCRIPT_DIR/config.conf

# python -u runner.py --config-file $1

# for config_file_seed in $1/*
# 	do
# 		echo submitted ${config_file_seed}
# 		python -u runner.py --config-file $config_file_seed
# 	done
source ~/.bashrc
conda activate mvenv

bbo_runner_path="/home/jhaa/NASLib/naslib/runners/bbo"
our_config="/home/jhaa/NASLib/naslib/configs/bbo/configs_cpu/nasbench101/cifar10/bananas/config_0/seed_0.yaml"
python -u -m debugpy --listen 0.0.0.0:$PORT --wait-for-client $bbo_runner_path/runner.py 
# --config-file $our_config
# echo $COMMAND;
# eval $COMMAND;

echo "DONE";
echo "Finished at $(date)";
conda deactivate 