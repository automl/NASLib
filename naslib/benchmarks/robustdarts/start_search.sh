#!/bin/bash

datasets="cifar10 cifar100 svhn"
spaces="s1 s2 s3 s4"

for d in $datasets; do
	for s in $spaces; do
		sbatch -J ${s}_${d} DARTS_search.sh $s $d
		echo submmited job $s $d
	done
done

