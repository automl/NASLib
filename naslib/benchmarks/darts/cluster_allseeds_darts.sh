#!/bin/bash
#SBATCH -p bosch_gpu-rtx2080 #ml_gpu-rtx2080     # bosch_gpu-rtx2080    #alldlc_gpu-rtx2080     # partition (queue)
#SBATCH --gres=gpu:1          # reserves one GPU
#SBATCH -o logs/%x.%A.%N.out       # STDOUT  %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%x.%A.%N.err       # STDERR  %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J darts-darts        # search space - algorithm
#SBATCH -a 1-4 # array size

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

###################
# commands here

# Activate virtual env so that run_experiment can load the correct packages
source /home/zelaa/NASLib/nl-venv/bin/activate
python runner.py --config-file config.yaml --optimizer darts --seed $SLURM_ARRAY_TASK_ID

#gpu_counter=1

#for seed in {1..16}; do
#  # Job to perform
#  if [ $gpu_counter -eq $SLURM_ARRAY_TASK_ID ]; then
#    #echo "Welcome $seed times"
#    #sleep 1
#    python runner.py --config-file config.yaml --optimizer darts --seed ${seed}
#    exit $?
#  fi
#  let gpu_counter+=1
#done



#
###################

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
