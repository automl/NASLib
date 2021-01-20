#!/bin/bash

optimizers="oneshot rsws"
space="darts nasbench201"

for s in $space
do
	for o in $optimizers
	do
		sbatch -J ${s}\_${o} slurm_job.sh $s $o
	done
done

