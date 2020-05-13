#!/bin/bash

datasets="cifar10 cifar100 svhn"
opt="DARTSOptimizer PCDARTSOptimizer"

for o in $opt; do
	for d in $datasets; do
		sbatch --bosch -J sdarts_${o}_${d} DARTS_eval.sh $d $o
		echo submmited job $d $o
		sleep 2
	done
done

