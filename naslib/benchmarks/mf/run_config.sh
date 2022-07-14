#!/bin/bash
directory=/path/to/configs

index=$1
# selects config by index given from SLURM scheduler
config_file_seed=$(ls -d $directory/* | sed "${index}q;d")

/home/gernel/miniconda3/bin/python -u ~/work/NASLib/naslib/benchmarks/mf/runner.py --config-file $config_file_seed
