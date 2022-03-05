#!/bin/bash

predictors=(bananas mlp lgb gcn bonas xgb ngb rf dngo \
 	bohamiann bayes_lin_reg seminas nao gp sparse_gp var_sparse_gp)

for predictor in ${predictors[@]}
do
    sbatch -J ${predictor} slurm_job.sh $predictor
done

