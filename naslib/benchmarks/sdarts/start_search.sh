#!/bin/bash

datasets="cifar10 cifar100 svhn"
opt="GDASOptimizer"

for o in $opt; do
	for d in $datasets; do
		sbatch --bosch -J sdarts_${o}_${d} DARTS_search.sh $d $o
		echo submmited job $d $o
		sleep 2
	done
done

