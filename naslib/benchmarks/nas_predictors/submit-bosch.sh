#!/bin/bash

#predictors=(bananas feedforward gbdt gcn bonas xgb ngb rf dngo \
# 	bohamiann bayes_lin_reg seminas gp sparse_gp var_sparse_gp)
predictors=(bananas feedforward gbdt bonas xgb ngb rf dngo \
 	bohamiann bayes_lin_reg gp sparse_gp var_sparse_gp)

for predictor in ${predictors[@]}
do
    sbatch -J nb301-${predictor} slurm_job-nb301.sh $predictor
    #sbatch --bosch -J nb101-${predictor} slurm_job-nb101.sh $predictor
done

