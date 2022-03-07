#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080,ml_gpu-rtx2080 #ml_gpu-rtx2080     # bosch_gpu-rtx2080    #alldlc_gpu-rtx2080     # partition (queue)
#SBATCH --gres=gpu:1          # reserves one GPU
#SBATCH -o logs_oneshot_eval/%x.%A-%a.%N.out       # STDOUT  %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs_oneshot_eval/%x.%A-%a.%N.err       # STDERR  %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -D .
#SBATCH -a 1 # array size

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

# Activate virtual env so that run_experiment can load the correct packages
source activate python37
python oneshot_runner.py --config-file nas_predictor_config.yaml \
	--model-path "run_epochs_size/${3}_${4}/cifar10/nas_predictors/${1}/${2}/$SLURM_ARRAY_TASK_ID/search/model_final.pth" \
	search_space $1 optimizer $2 search.predictor_type $2 \
	seed $SLURM_ARRAY_TASK_ID search.seed $SLURM_ARRAY_TASK_ID \
	search.train_portion 0.$3 search.epochs $4 \
	out_dir run_epochs_size-eval/$3\_$4


end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
