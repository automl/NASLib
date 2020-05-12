#!/bin/bash

datasets="cifar10 cifar100 svhn"
opt="DARTSOptimizer GDASOptimizer PCDARTSOptimizer"

for o in $opt; do
	for d in $datasets; do
		sbatch --bosch -J ${o}_${d} DARTS_search.sh $d $o
		echo submmited job $d $o
		sleep 2
	done
done

