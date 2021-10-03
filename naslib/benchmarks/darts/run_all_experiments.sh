#!/bin/bash

optimizers=(darts gdas drnas)
search_spaces=(nasbench101 nasbench201 darts)


for opt in "${optimizers[@]}"
do
	for search_space in "${search_spaces[@]}"
	do
		echo run_exp.sh $opt $search_space
		sbatch run_exp.sh $opt $search_space
	done
done
