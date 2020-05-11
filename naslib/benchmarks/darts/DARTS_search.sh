#!/bin/bash
#
# submit to the right queue
#SBATCH -p bosch_gpu-rtx2080
#SBATCH --gres gpu:1
#SBATCH -a 1
#
# the execution will use the current directory for execution (important for relative paths)
#SBATCH -D .
#
# redirect the output/error to some files
#SBATCH -o ./logs/%A_%a.o
#SBATCH -e ./logs/%A_%a.e
#
#

source activate tensorflow-stable
python runner.py --dataset $1 --optimizer $2

