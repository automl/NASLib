#!/bin/bash

datasets="cifar10 cifar100 svhn"
opt="DARTSOptimizer GDASOptimizer PCDARTSOptimizer"
spaces="s1 s2 s3 s4"

for o in $opt; do
	for d in $datasets; do
		for s in $spaces; do
			sbatch -J ${s}_${d}_${o} --bosch DARTS_eval.sh $s $d $o
			echo submmited job $s $d $o
			sleep 3
		done
	done
done

