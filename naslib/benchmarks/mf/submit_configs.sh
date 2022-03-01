#!/bin/bash
# define variable names
workspace=/work/dlclarge2/gernel-naslib-experiments


#SBATCH -p mlhiwidlc_gpu-rtx2080 #bosch_gpu-rtx2080    #bosch_cpu-cascadelake  # partition (queue)
#SBATCH -t 0-1:00           # time (D-HH:MM)
#SBATCH -o /work/dlclarge2/gernel-naslib-experiments/log/%A.%N.out       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e /work/dlclarge2/gernel-naslib-experiments/log/%A.%N.err       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J naslib-bbo-rs            # sets the job name. 
#SBATCH --mem=40G
#SBATCH -c=8
#SBATCH --chdir=/work/dlclarge2/gernel-naslib-experiments/NASLib
# Print some information about the job to STDOUT

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

config_dir=$workspace/NASLib/naslib/benchmarks/mf/configs/nasbench201/cifar10/sh
if [[ $# -eq 0 ]] ; then
    echo 'some message'
    exit 0
fi
start=$1
end=$2
echo "Starting at config $start"
# python -u runner.py --config-file $1
i=-1
for config in $config_dir/*
    do
    for config_file_seed in $config/*
        do
            ((i=i+1))
            if [[ "$i" -ge "$start" ]] && [[ "$i" -lt "$end" ]]; then # &&  $i -gt $end ]] ; then
		echo $i
            	echo submitted ${config_file_seed}
		echo "/home/gernel/miniconda3/bin/python -u $workspace/NASLib/naslib/benchmarks/mf/runner.py --config-file $config_file_seed"
	    fi
        done
    done
 
# echo $COMMAND;
# eval $COMMAND;

echo "DONE";
echo "Finished at $(date)";
