#!/bin/bash

#580,423,340,273,204

datasets="cifar10 cifar100 svhn"
spaces="s1 s2 s3 s4"
opt="DARTSOptimizer PCDARTSOptimizer"

for o in $opt; do
	for d in $datasets; do
		for s in $spaces; do
			sbatch -J ${s}_${d}_${o} --bosch DARTS_eval.sh $s $d $o
			echo submmited job $s $d $o
			sleep 2
		done
	done
done

