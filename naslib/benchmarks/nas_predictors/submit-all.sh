#!/bin/bash

#nb101
#predictors101=(bananas mlp lgb gcn xgb ngb rf dngo \
	#bohamiann bayes_lin_reg seminas nao gp sparse_gp var_sparse_gp)

#nb201
#predictors201=(bananas mlp lgb gcn bonas xgb ngb rf dngo \
 	#bohamiann bayes_lin_reg seminas nao gp sparse_gp var_sparse_gp)
predictors201=(ngb_hp omni)

#nb301
#predictors301=(bananas mlp lgb bonas xgb ngb rf dngo \
# 	bohamiann bayes_lin_reg gp sparse_gp var_sparse_gp nao)

#for predictor in ${predictors101[@]}
#do
    #sbatch -J 101-${predictor} slurm_job-nb101.sh $predictor
#done

for predictor in ${predictors201[@]}
do
    #sbatch -J 201-${predictor} slurm_job-nb201-c10.sh $predictor
    #sbatch -J c100-201-${predictor} slurm_job-nb201-c100.sh $predictor
    #sbatch -J imnet-201-${predictor} slurm_job-nb201-imagenet.sh $predictor
    sbatch -J imnet-201-${predictor} slurm_job-imgnet.sh $predictor
done

#for predictor in ${predictors301[@]}
#do
    #sbatch -J 301-${predictor} slurm_job-nb301.sh $predictor
#done

