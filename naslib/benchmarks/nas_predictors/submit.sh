#!/bin/bash

predictors=(bananas feedforward gbdt gcn bonas xgb ngb rf dngo \
	bohamiann bayes_lin_reg seminas gp sparse_gp var_sparse_gp)

for predictor in ${predictors[@]}
do
    sbatch -J ${predictor} slurm_job.sh $predictor
done

