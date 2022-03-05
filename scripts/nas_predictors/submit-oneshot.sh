#!/bin/bash

optimizers="oneshot rsws"
space="darts nasbench201"
portion="25 5 75 9"
epochs="25 50 100 150"

for s in $space
do
	for o in $optimizers
	do
		for p in $portion
		do
			for e in $epochs
			do
				sbatch -J ${s}\_${o}\_$p\_$e oneshot_eval.sh $s $o $p $e
				echo $s $o 0.$p\_$e
			done
		done
	done
done
#for s in $space
#do
	#for o in $optimizers
	#do
		#for p in $portion
		#do
			#for e in $epochs
			#do
				#scancel -n ${s}\_${o}\_$p\_$e
			#done
		#done
	#done
#done
