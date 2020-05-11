#!/bin/bash

datasets="cifar10 cifar100 svhn"

for d in $datasets; do
	sbatch -J darts_${d} DARTS_search.sh $d
	echo submmited job $d
done

