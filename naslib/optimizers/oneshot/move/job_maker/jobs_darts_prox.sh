#!/bin/bash

#SBATCH -p alldlc_gpu-rtx2080 #testdlc_gpu-rtx2080 #mlhiwi_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/darts_10_%A_%a.out
#SBATCH --error=slurm/darts_10_%A_%a.err



echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

python runner.py --config-file config.yaml seed $1 search.epochs $2 search.warm_start_epochs $3 search.instantenous $4 dataset $5 search.masking_interval $6 search_space $7 search.train_portion $8 search.batch_size $9 search.data_size ${10} out_dir ${11}

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime